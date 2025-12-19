import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_xy(filename):
    """Load only first two columns, skipping lines beginning with '#'."""
    data = []
    with open(filename) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split()
            # use only first two columns as floats
            data.append([float(cols[0]), float(cols[1])])
    data.append(data[0])  # close the path from the last to the 1st city
    return np.array(data)

def load_polygons(filename):
    """Load longitude/latitude polygons separated by blank lines."""
    polygons = []
    current = []

    with open(filename) as f:
        for line in f:
            line = line.strip()

            # Blank line = end of current polygon
            if not line:
                if current:
                    polygons.append(np.array(current))
                    current = []
                continue

            # Parse longitude/latitude
            parts = line.split()
            lon = float(parts[0])
            lat = float(parts[1])
            current.append([lon, lat])

    # Add last polygon if missing trailing blank line
    if current:
        polygons.append(np.array(current))

    return polygons

def make_plot(infile, optfile=None, region="NA"):
    '''
    infile: (required) a list of cities
    optfile: a list of cities ordered for optimized route
    region: area of the globe to plot
            NA = North America
            World = the whole world
    '''

    # Load data
    polygons = load_polygons("world_50m.dat")
    #polygons = load_polygons("world.dat") # low res version
    cities_orig = load_xy(infile)
    if optfile: 
        cities_out = load_xy(optfile)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot world map (outline)
    for poly in polygons:
        ax.plot(poly[:,0], poly[:,1], color="black", lw=0.8)

    # Plot original city order (thin red line)
    orig_line, = ax.plot(cities_orig[:,0], cities_orig[:,1], lw=1, color="red", alpha=0.5, label="Original Route")  # new label for legend

    # Plot salesman path (lines + points)
    if optfile:
        sa_line, = ax.plot(cities_out[:,0], cities_out[:,1],
                           lw=2, color="blue", marker='o', markersize=3, label="Optimized Route")  # new label for legend

    # Add legend box in top-right corner
    ax.legend(loc='upper right', frameon=True)  # new comment: show what red and blue lines represent

    # Labels
    ax.set_title("Plot of Salesman's Cities")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Axis ranges
    if region == "NA":
        ax.set_xlim(-180, -60)
        ax.set_ylim(10, 75)
    elif region == "World":
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

    # Remove top + right border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save plot
    plotfile = infile.split('.')[0] + '.pdf'
    plt.savefig(plotfile, format='pdf', facecolor='white')
    print(f"Plot saved as {plotfile}")

    print('Close plot or "^C" to exit')
    try:
        plt.show()
    except KeyboardInterrupt:
        print("Interrupted with Ctrl-C, closing plot and exiting...")
        plt.close('all')


def usage():
    print('usage:')
    print('python routeplot.py cities.dat [cities2.dat] -w plot the whole world')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Route plotter')
    parser.add_argument('paths', nargs='*', help='paths to plot: original [optimized]')
    parser.add_argument("-w", action='store_true', help="plot the whole world, default North America only")
    args = parser.parse_args()

    if len(args.paths) < 1:
        print("At least one input file needed")
        usage()
        exit(1)

    cities = args.paths[0]
    cities2 = None
    if len(args.paths) > 1:
        cities2 = args.paths[1]

    region = "World" if args.w else "NA"
    make_plot(cities, cities2, region)

