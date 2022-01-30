from statistics import median, stdev, mean

import matplotlib.pyplot as plt
import sys

def read_data(model_name : str):
    with open(f"scores/scores_{model_name}") as file:
        content = file.read()

    numbers = content.split()
    numbers = [float(number) for number in numbers]

    return numbers


def draw_plot(data):
    plt.plot(data)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.show()


def print_statistics(data):
    print(f'Number of episodes : {len(data)}')
    print(f'Maximum score : {max(data)} . Minimum score : {min(data)}')

    mean_score = median(data)

    print(f'Mean score : {mean(data)} . Median score : {median(data)}')
    print(f'Standard deviation : {stdev(data)}')

    over_mean = len(list(filter(lambda x: x > mean_score, data)))
    under_mean = len(data) - over_mean
    print(f"Number of scores over mean : {over_mean}. Under : {under_mean}")

    over_1000 = len(list(filter(lambda x: x > 1000,data)))
    print(f"Over 1000 : {over_1000}")


def main():
    model_name = sys.argv[1]
    data = read_data(model_name)
    print_statistics(data)
    draw_plot(data)


if __name__ == "__main__":
    main()