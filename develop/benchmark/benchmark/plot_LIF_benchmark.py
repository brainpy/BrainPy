import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pdb

#from experiments import ROOT_DIR

figure_path = os.path.abspath('.')
benchmark_path = os.path.abspath('.')

if not os.path.isdir(benchmark_path):
    os.makedirs(benchmark_path)


def main(start=100, stop=1000, step=100, time=1000, interval=100):
    name = f"benchmark_LIF_sim_{start}_{stop}_{step}_{time}"
    f = os.path.join(benchmark_path, name + ".csv")
    df = pd.read_csv(f, index_col=0)
    
    #fig,ax = plt.subplots()
    plt.plot(df["BindsNET_cpu"], label="BindsNET", linestyle="-", color="b")
    plt.plot(df["BRIAN2"], label="BRIAN2", linestyle="--", color="r")
    plt.plot(df["PyNEST"], label="PyNEST", linestyle="--", color="y")
    plt.plot(df["ANNarchy_cpu"], label="ANNarchy", linestyle="--", color="m")
    plt.plot(df["Nengo"], label="Nengo", linestyle="--", color="c")
    plt.plot(df["BrainPy_cpu"], label="BrainPy", linestyle="--", color="g")

    plt.title("Benchmark comparison of SNN simulation libraries")
    plt.xticks(range(0, stop + interval, interval))
    plt.xlabel("Number of input / output neurons")
    plt.ylabel("Simulation time (seconds)")
    plt.legend(loc=1, prop={"size": 5})
    plt.yscale("log")
    plt.legend()
    plt.show()
    #plt.savefig(os.path.join(figure_path, name + ".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--time", type=int, default=1000)
    parser.add_argument("--interval", type=int, default=1000)
    args = parser.parse_args()

    main(
        start=args.start,
        stop=args.stop,
        step=args.step,
        time=args.time,
        interval=args.interval
    )
