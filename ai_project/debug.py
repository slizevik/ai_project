def scale_values(values: list[float], factor: float) -> list[float]:
    return [value + factor for value in values]


def count_values(values: list[float]) -> int:
    return len(values)


if __name__ == "__main__":
    print(scale_values([1.0, 2.0, 3.0], 2.0))
