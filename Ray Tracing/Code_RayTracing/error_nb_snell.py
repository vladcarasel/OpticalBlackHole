from plots_ray_tracing_snell import make_figure_6, make_figure_6_full


def main():
    print("Generating Fig. 6: ray-tracing error analysis...")
    make_figure_6()
    make_figure_6_full()
    print("Saved Fig. 6 outputs in results/ray_tracing/")


if __name__ == "__main__":
    main()
