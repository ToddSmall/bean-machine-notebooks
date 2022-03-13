import beanmachine.ppl as bm
import torch
import torch.distributions as dist


def load_data() -> dict[str, torch.Tensor]:
    x_obs = torch.tensor(
        [0, 9, 14, 15, 20, 21, 30, 35, 40, 41, 42, 43, 54, 56, 67, 69, 88],
        dtype=torch.float32,
    )
    y_obs = torch.tensor(
        [33, 34, 34, 37, 37, 44, 48, 49, 53, 49, 50, 48, 56, 60, 61, 63, 71],
        dtype=torch.float32,
    )
    return {"x_obs": x_obs, "y_obs": y_obs}


class LinearRegressionModel:
    def __init__(self, x_obs: torch.Tensor) -> None:
        self._x_obs = x_obs

    @bm.random_variable
    def beta_0(self) -> dist.Distribution:
        return dist.Normal(0, 10)

    @bm.random_variable
    def beta_1(self) -> dist.Distribution:
        return dist.Normal(0, 10)

    @bm.random_variable
    def sigma(self) -> dist.Distribution:
        return dist.Gamma(1, 1)

    @bm.random_variable
    def y(self) -> dist.Distribution:
        return dist.Normal(self.beta_0() + self.beta_1() * self._x_obs, self.sigma())


def classic_linear_regression() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = load_data()
    X = torch.vstack([torch.ones(len(data["x_obs"])), data["x_obs"]]).T
    return torch.linalg.lstsq(X, data["y_obs"], rcond=None)


def main() -> None:
    bm.seed(883344)
    num_samples = 4000
    num_chains = 4
    data = load_data()
    model = LinearRegressionModel(x_obs=data["x_obs"])
    samples = bm.inference.BMGInference().infer(
        queries=[model.beta_0(), model.beta_1(), model.sigma()],
        observations={model.y(): data["y_obs"]},
        num_samples=num_samples,
        num_chains=num_chains,
    )
    print(bm.Diagnostics(samples).summary())


if __name__ == "__main__":
    main()
