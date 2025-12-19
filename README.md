Starter code and data for traveling salesman problem


Files in this directory:

* datareader.cpp : example code to read in the data files (use Makefile)
* datareader.py  : example code to read in the data files
* cities23.dat : list of coordinates for 23 cities in North America
* cities150.dat : 150 cities in North America
* cities1k.dat : 1207 cities in North America
* cities2k.dat : 2063 cities around the world
* routeplot.py : code to plot the globe and salesman's path<br>
usage:<br>
python routeplot.py cities.dat [cities2.dat] -r [="NA"],"World"'<br>
NA = North America, World = Mercator projection of the whole earth
* earth.C : (just for fun) plotting the globe in ROOT



# Traveling Salesman Problem – Simulated Annealing

## Build and Run Instructions

This project is written in Python and does not require compilation.

Make sure you are using the `phys56xx` environment and that the following files are present:

- sales.py  
- routeplot.py  
- cities23.dat  
- cities150.dat  
- cities1k.dat  
- cities2k.dat  

Run the program using:

python sales.py <input_file> <output_file>

Examples:

python sales.py cities23.dat route23.dat
python sales.py cities150.dat route150.dat
python sales.py cities1k.dat route1k.dat
python sales.py cities2k.dat route2k.dat

The program writes an optimized route file, generates a PDF plot of the route, and produces an annealing schedule plot.

To generate the route plots:

python routeplot.py <input_file> <route_file> <output_pdf>

For the full-world plot (e.g., 2k cities dataset), use the -w option:

python routeplot.py cities2k.dat route2k.dat -w

# Results

| Dataset    | Original Length (km) | Nearest Neighbor (km) | Simulated Annealing (km) | Time (s) |
|-----------|--------------------:|---------------------:|------------------------:|---------:|
| cities23  | 38,963.19           | 13,859.39            | 13,404.55               | 37.42    |
| cities150 | 317,298.65          | 56,179.86            | 49,765.03               | 45.89    |
| cities1k  | 732,177.74          | 119,212.88           | 102,685.70              | 113.29   |
| cities2k  | 10,187,617.64       | 355,493.95           | 355,470.95              | 190.33   |

> All distances are **round-trip distances** in kilometers.


##  Plots

### Route Plots (PDF)

- `cities23.pdf`  
- `cities150.pdf`  
- `cities1k.pdf`  
- `cities2k.pdf` (use the `-w` option)

### Annealing Schedules (Distance vs Temperature)

- `an23.png`  
- `an150.png`  
- `an1.png`  
- `an2.png`  

> Each plot shows the total route distance as a function of temperature during the simulated annealing process.

##  Updates to `routeplot.py`

- **Legend added:** The plot now includes a legend in the top-left corner to indicate which line corresponds to which route:  
  - ** Red line:** Original city order  
  - ** Blue line:** Optimized route (Simulated Annealing)

