import ray

from parameter_server import ParameterServer


def main():
    ray.init(address="auto")
    ray.get(ParameterServer.remote().run_training.remote())


if __name__ == "__main__":
    main()
