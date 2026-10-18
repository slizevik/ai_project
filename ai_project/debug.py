def scale_values(values: list[float], factor: float) -> list[float]:
    return [factor * value for value in values]


def total_values(values: list[float]) -> float:
    return sum(values)


def count_values(values: list[float]) -> int:
    return len(values)


if __name__ == "__main__":
    print(scale_values([1.0, 2.0, 3.0], 2.0))
