
# Gradient Coding — K3s / KubeRay / CUDA

Экспериментальная система распределённого обучения ResNet-18 на CIFAR-10 с поддержкой Classic Gradient Coding, CCE-GC-CR/FR и PGC-CR/FR. Вычисления реализованы на PyTorch/CUDA, распределённое выполнение — на Ray Actors. Эксперименты регистрируются в MLflow.

## Архитектура стенда

- Один физический сервер с **openSUSE Leap 15.6**, NVIDIA GeForce RTX 2070 (8 ГБ VRAM).
- **K3s** и **KubeRay**: один Ray Head (ParameterServer, CPU) и шесть Ray Worker Pods (GCWorker, CUDA).
- **NVIDIA Device Plugin time-slicing** предоставляет шесть логических `nvidia.com/gpu` на одной физической видеокарте. Это не шесть независимых GPU.
- **SILO S3** (`silo` namespace): `gc-datasets` (CIFAR-10) и `mlflow-artifacts` (артефакты запусков); PVC 20 GiB.
- **MLflow 3.6 + PostgreSQL 16** (`default` namespace): регистрация запусков и метрик; PVC PostgreSQL 5 GiB.
- Начальные данные скачиваются initContainer в `emptyDir` каждого Ray Pod. Промежуточные JSONL хранятся на Ray Head в `/data/results`, долговременные копии — в MLflow/S3.

## 1. Требования и подготовка ОС

Для развёртывания на новом узле после публикации репозитория сначала выполните `git clone <URL_репозитория>` и перейдите в каталог проекта. Необходимы `git`, `curl`, `docker`, `helm`, `python3`, `kubectl`, рабочий драйвер NVIDIA и доступ к интернету для первоначальной установки. До продолжения должна выполняться команда:

```bash
nvidia-smi
```

Для OpenSUSE установка NVIDIA Container Toolkit (если ещё не установлен):

```bash
sudo zypper ar https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
sudo zypper --gpg-auto-import-keys install -y nvidia-container-toolkit
```

Установка K3s на чистом узле:

```bash
curl -sfL https://get.k3s.io | sh -
sudo systemctl enable --now k3s
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$(id -u):$(id -g)" ~/.kube/config
chmod 600 ~/.kube/config
kubectl get nodes
```

После установки NVIDIA runtime (или если он был установлен уже после K3s) перезапустить K3s и проверить RuntimeClass:

```bash
sudo systemctl restart k3s
kubectl get runtimeclass nvidia
sudo grep -i nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml
```

K3s использует собственный **containerd**, поэтому сборка образа в Docker на хосте ещё не делает образ доступным Kubernetes.

## 2. NVIDIA Device Plugin: шесть логических GPU

Если NVIDIA Device Plugin уже установлен (или установлен NVIDIA GPU Operator), не разворачивайте второй экземпляр плагина. Для чистого стенда с standalone NVIDIA Device Plugin используется `k8s/nvidia-device-plugin-values.yaml`:

```yaml
config:
  default: time-slicing
  map:
    time-slicing: |-
      version: v1
      flags:
        migStrategy: none
      sharing:
        timeSlicing:
          renameByDefault: false
          failRequestsGreaterThanOne: true
          resources:
            - name: nvidia.com/gpu
              replicas: 6
```

Установка:

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade --install nvdp nvdp/nvidia-device-plugin \
  -n nvidia-device-plugin --create-namespace \
  -f k8s/nvidia-device-plugin-values.yaml --wait

kubectl describe node "$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')" | grep 'nvidia.com/gpu'
```

В `Allocatable` ожидаются шесть логических GPU-ресурсов. Time-slicing предоставляет совместный доступ к одной видеокарте без изоляции видеопамяти и гарантированной производительности.

## 3. KubeRay Operator

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

helm upgrade --install kuberay-operator kuberay/kuberay-operator \
  --namespace ray-system --create-namespace --version 1.7.0 --wait

kubectl get pods -n ray-system
```

KubeRay управляет ресурсами `RayCluster` и `RayJob` внутри существующего Kubernetes-кластера.

## 4. SILO: S3-хранилище

`k8s/silo-values.yaml` использует режим `standalone`, `existingSecret: silo-root`, PVC `local-path` 20 GiB и сервисы на портах 9000/9001.

```bash
kubectl create namespace silo --dry-run=client -o yaml | kubectl apply -f -

read -rsp 'SILO admin password: ' SILO_PASSWORD; echo
kubectl -n silo create secret generic silo-root \
  --from-literal=rootUser=gcadmin \
  --from-literal=rootPassword="$SILO_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -
unset SILO_PASSWORD

git clone --branch RELEASE.2026-11-12T00-00-00Z --depth 1 \
  https://github.com/pgsty/silo.git /tmp/silo-src

helm upgrade --install silo /tmp/silo-src/helm/silo \
  -n silo -f k8s/silo-values.yaml --wait

kubectl -n silo get pods,svc,pvc
```

Создать S3-бакеты и включить версионирование датасета:

```bash
kubectl -n silo exec deploy/silo -- sh -c \
  'mcli alias set local http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"'

kubectl -n silo exec deploy/silo -- mcli mb --ignore-existing local/gc-datasets
kubectl -n silo exec deploy/silo -- mcli mb --ignore-existing local/mlflow-artifacts
kubectl -n silo exec deploy/silo -- mcli version enable local/gc-datasets
```

Создать отдельную учётную запись только для чтения обучающих данных:

```bash
read -rsp 'S3 dataset reader password: ' S3_READ_PASSWORD; echo

kubectl -n silo exec deploy/silo -- mcli admin user add local gc-reader "$S3_READ_PASSWORD"
kubectl -n silo exec deploy/silo -- mcli admin policy attach local readonly --user gc-reader

kubectl create secret generic gc-s3-reader \
  --from-literal=AWS_ACCESS_KEY_ID=gc-reader \
  --from-literal=AWS_SECRET_ACCESS_KEY="$S3_READ_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

unset S3_READ_PASSWORD
```

Создать отдельную учётную запись для записи MLflow-артефактов (для лабораторного стенда здесь используется стандартная политика `readwrite`):

```bash
read -rsp 'S3 MLflow password: ' S3_MLFLOW_PASSWORD; echo

kubectl -n silo exec deploy/silo -- mcli admin user add local gc-mlflow "$S3_MLFLOW_PASSWORD"
kubectl -n silo exec deploy/silo -- mcli admin policy attach local readwrite --user gc-mlflow

kubectl create secret generic gc-s3-mlflow \
  --from-literal=AWS_ACCESS_KEY_ID=gc-mlflow \
  --from-literal=AWS_SECRET_ACCESS_KEY="$S3_MLFLOW_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

unset S3_MLFLOW_PASSWORD
```

Секреты находятся в Kubernetes и **не хранятся в Git**.

## 5. Загрузка CIFAR-10 в S3

Скачайте официальный архив `cifar-10-python.tar.gz` в произвольную папку вне репозитория: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>.

Для первоначальной загрузки с хостового Linux используйте `boto3`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install boto3
```

В отдельном терминале откройте порт S3 API:

```bash
kubectl -n silo port-forward svc/silo 9000:9000
```

В основном терминале задайте путь к архиву и временные учётные данные администратора SILO:

```bash
CIFAR_ARCHIVE=/absolute/path/to/cifar-10-python.tar.gz
export AWS_ACCESS_KEY_ID=gcadmin

read -rsp 'SILO admin password: ' AWS_SECRET_ACCESS_KEY; echo
export AWS_SECRET_ACCESS_KEY

python3 - "$CIFAR_ARCHIVE" <<'PY'
import sys
import boto3
from botocore.config import Config

client = boto3.client(
    's3',
    endpoint_url='http://127.0.0.1:9000',
    region_name='us-east-1',
    config=Config(s3={'addressing_style': 'path'})
)

client.upload_file(
    sys.argv[1],
    'gc-datasets',
    'cifar10/v1/cifar-10-python.tar.gz'
)
print('Uploaded CIFAR-10 to gc-datasets')
PY

unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
```

Создайте ConfigMap с адресом S3-объекта и контрольной суммой архива:

```bash
CIFAR_SHA256=$(sha256sum "$CIFAR_ARCHIVE" | awk '{print $1}')

kubectl create configmap gc-s3-config \
  --from-literal=S3_ENDPOINT_URL=http://silo.silo.svc.cluster.local:9000 \
  --from-literal=S3_BUCKET=gc-datasets \
  --from-literal=S3_KEY=cifar10/v1/cifar-10-python.tar.gz \
  --from-literal=CIFAR10_SHA256="$CIFAR_SHA256" \
  --dry-run=client -o yaml | kubectl apply -f -
```

При создании Ray Pods initContainer `/app/s3_dataset.py` загрузит архив в `emptyDir`, проверит SHA-256 и распакует данные в `/data/cifar10`.

## 6. Сборка и загрузка контейнерных образов

`Dockerfile` проекта должен собирать **CUDA-версию**, а не старую CPU-версию. В образе Ray должны присутствовать `torch==2.8.0` с CUDA 12.6, `torchvision==0.23.0`, `boto3` и клиент `mlflow==3.6.0`. Базовый образ — `rayproject/ray:2.58.0-py311-cpu` с последующей установкой CUDA-колёс PyTorch.

Отдельный `Dockerfile.mlflow`:

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==3.6.0 'psycopg2-binary>=2.9,<3' 'boto3>=1.35,<2'
```

Образы собираются через Docker и импортируются во встроенный containerd K3s:

```bash
docker build -t gc-lab-v1:cuda -f Dockerfile .
docker build -t gc-mlflow:3.6.0 -f Dockerfile.mlflow .

docker save -o /tmp/gc-lab-v1-cuda.tar gc-lab-v1:cuda
sudo k3s ctr -n k8s.io images import /tmp/gc-lab-v1-cuda.tar

docker save -o /tmp/gc-mlflow.tar gc-mlflow:3.6.0
sudo k3s ctr -n k8s.io images import /tmp/gc-mlflow.tar

sudo k3s ctr -n k8s.io images ls | grep -E 'gc-lab-v1|gc-mlflow'

rm -f /tmp/gc-lab-v1-cuda.tar /tmp/gc-mlflow.tar
```

В манифестах для локальных образов используется `imagePullPolicy: Never`.

## 7. PostgreSQL и MLflow

Создайте Secret для PostgreSQL:

```bash
read -rsp 'PostgreSQL mlflow password: ' POSTGRES_PASSWORD; echo

kubectl create secret generic mlflow-db \
  --from-literal=password="$POSTGRES_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

unset POSTGRES_PASSWORD
```

Разверните сервер и базу данных в namespace `default`:

```bash
kubectl apply -f k8s/mlflow.yaml
kubectl rollout status deploy/postgres --timeout=300s
kubectl rollout status deploy/mlflow --timeout=300s
kubectl get pods,svc,pvc
```

MLflow сохраняет метаданные и метрики в PostgreSQL, а `config.json`, `step_metrics.jsonl` и `epoch_metrics.jsonl` — в бакет `mlflow-artifacts` посредством S3 API.

## 8. Запуск RayCluster и RayJob

Убедитесь, что `k8s/raycluster.yaml` соответствует текущему CUDA-стенду (Head + 6 GPU Worker Pods, image `gc-lab-v1:cuda`, `runtimeClassName: nvidia`):

```bash
kubectl apply -f k8s/raycluster.yaml
kubectl get rayclusters
kubectl get pods -l ray.io/cluster=gc-lab-v1 -o wide

kubectl wait --for=condition=Ready pods \
  -l ray.io/cluster=gc-lab-v1 --timeout=600s
```

В `k8s/rayjob.yaml` задайте `GC_SCHEME`, `NUM_WORKERS`, `GC_STRAGGLERS`, `CCE_M`, `TRAIN_LIMIT`, `BATCH_SIZE`, `EPOCHS`, `LR` и другие параметры. Поддерживаются `classic`, `cce_gc_cr`, `cce_gc_fr`, `pgc_cr`, `pgc_fr`. Для `n=6`, `s=1`, `m=2` доступны обе CCE-схемы; другие комбинации должны удовлетворять ограничениям выбранной схемы.

```bash
kubectl apply -f k8s/rayjob.yaml
kubectl get rayjobs -w
```

При повторном использовании того же `metadata.name` сначала удалите завершённый RayJob:

```bash
kubectl delete rayjob gc-lab-v1-run --ignore-not-found
kubectl apply -f k8s/rayjob.yaml
```

Для нового независимого эксперимента рекомендуется использовать новое имя RayJob, чтобы сохранять Kubernetes-историю запусков.

## 9. Мониторинг и получение результатов

Ray Dashboard:

```bash
kubectl port-forward svc/gc-lab-v1-head-svc 8265:8265
# http://localhost:8265
```

MLflow UI (в другом терминале):

```bash
kubectl port-forward svc/mlflow 5000:5000
# http://localhost:5000
```

Проверка статуса и логов задания:

```bash
kubectl get rayjobs

HEAD=$(kubectl get pods -l ray.io/node-type=head \
  -o jsonpath='{.items[0].metadata.name}')

JOB_ID=$(kubectl get rayjob gc-lab-v1-run \
  -o jsonpath='{.status.jobId}')

kubectl exec "$HEAD" -c ray-head -- ray job logs \
  --address http://127.0.0.1:8265 "$JOB_ID"
```

Промежуточные JSONL на Head:

```bash
kubectl exec "$HEAD" -c ray-head -- ls -lah /data/results
```

Они находятся во временном `emptyDir`. Долговременные артефакты нужно получать из MLflow/S3; удаление Ray Head Pod удалит его локальные файлы `emptyDir`.

## Замечание об экспериментальной среде

Шесть Ray Worker Pods являются отдельными Ray-узлами, однако используют один физический GPU посредством NVIDIA time-slicing. Их производительность зависит от конкуренции за вычислительные ресурсы и общую видеопамять. Стенд предназначен для проверки работы схем Gradient Coding, измерения характеристик обучения и взаимодействия компонентов, а не для имитации производительности шести независимых физических GPU.
