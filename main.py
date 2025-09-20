import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic, great_circle
import math
from itertools import product
import warnings
import time
import sys
import os
import shutil
import threading
from datetime import timedelta
from colorama import init, Fore, Style, Back

# Initialize colorama for cross-platform colored terminal output
init()

# Suppress warnings
warnings.filterwarnings("ignore")

EARTH_RADIUS = 6371000  # Earth radius in meters
EARTH_FLATTENING = 1/298.257223563  # WGS-84 flattening parameter

LOGO = r"""
 ________  ________  ________  ________  ________
|\   ____\|\   __  \|\   __  \|\   __  \|\   ___ \
\ \  \___|\ \  \|\  \ \  \|\  \ \  \|\  \ \  \_|\ \
 \ \  \    \ \  \\\  \ \  \\\  \ \   _  _\ \  \ \\ \
  \ \  \____\ \  \\\  \ \  \\\  \ \  \\  \\ \  \_\\ \
   \ \_______\ \_______\ \_______\ \__\\ _\\ \_______\
    \|_______|\|_______|\|_______|\|__|\|__|\|_______|



 ________ ___  ________   ________  _______   ________
|\  _____\\  \|\   ___  \|\   ___ \|\  ___ \ |\   __  \
\ \  \__/\ \  \ \  \\ \  \ \  \_|\ \ \   __/|\ \  \|\  \
 \ \   __\\ \  \ \  \\ \  \ \  \ \\ \ \  \_|/_\ \   _  _\
  \ \  \_| \ \  \ \  \\ \  \ \  \_\\ \ \  \_|\ \ \  \\  \|
   \ \__\   \ \__\ \__\\ \__\ \_______\ \_______\ \__\\ _\
    \|__|    \|__|\|__| \|__|\|_______|\|_______|\|__|\|__|
                          Lukáš Klíma
"""

# Spinner animation for loading
def spinner(text="Processing", delay=0.1, iterations=None, stop_event=None):
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    loops = 0
    while True:
        if stop_event and stop_event.is_set():
            break
        char = spinner_chars[i % len(spinner_chars)]
        sys.stdout.write(f'\r{text} {char}')
        sys.stdout.flush()
        time.sleep(delay)
        i += 1
        if iterations is not None:
            loops += 1
            if loops >= iterations:
                break
    sys.stdout.write('\r' + ' ' * (len(text) + 10) + '\r')
    sys.stdout.flush()

def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}{Style.RESET_ALL}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def print_header(text):
    print(f"\n{'═' * 60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{text.center(60)}{Style.RESET_ALL}")
    print(f"{'═' * 60}{Style.RESET_ALL}\n")

def print_result(label, value, color=Fore.GREEN):
    print(f"{label}: {color}{value}{Style.RESET_ALL}")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def vincenty_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance using Vincenty's formula via geopy's geodesic implementation.
    Falls back to great-circle if Vincenty fails.
    """
    try:
        return geodesic((lat1, lon1), (lat2, lon2), ellipsoid='WGS-84').meters
    except ValueError:
        return great_circle((lat1, lon1), (lat2, lon2)).meters

def trilateration_objective(coordinates, reference_points, measured_distances):
    """
    Objective function for optimization.
    Minimizes the sum of squared relative errors between measured and calculated distances.
    """
    lat, lon = coordinates
    errors = []

    for i, (ref_lat, ref_lon, distance) in enumerate(zip(reference_points["lat"],
                                                       reference_points["lon"],
                                                       measured_distances)):
        calculated_distance = vincenty_distance(lat, lon, ref_lat, ref_lon)
        relative_error = (calculated_distance - distance) / distance
        errors.append(relative_error)

    return np.sum(np.array(errors) ** 2)

def grid_search(reference_points, measured_distances, resolution=30):
    """
    Perform a global grid search to find good initial starting points for optimization.
    Returns multiple candidate points to be refined further.
    """
    lats = np.linspace(-90, 90, resolution)
    lons = np.linspace(-180, 180, resolution)

    best_candidates = []
    best_errors = []

    total_points = resolution * resolution
    print(f"{Fore.WHITE}Performing global grid search...{Style.RESET_ALL}")

    point_count = 0
    for lat, lon in product(lats, lons):
        error = trilateration_objective([lat, lon], reference_points, measured_distances)
        best_candidates.append((lat, lon))
        best_errors.append(error)

        point_count += 1
        if point_count % 10 == 0:
            progress_bar(point_count, total_points, prefix='Progress:', suffix='Complete', length=40)

    sorted_indices = np.argsort(best_errors)
    return [best_candidates[i] for i in sorted_indices[:5]]

def multi_stage_optimization(reference_points, measured_distances):
    """
    Multi-stage optimization approach:
    1. Global search using differential evolution
    2. Fine-tuning using L-BFGS-B from the best global solution
    """
    spinner("Running global optimization (differential evolution)", delay=0.1, iterations=10)

    bounds = [(-90, 90), (-180, 180)]
    result_global = differential_evolution(
        trilateration_objective,
        bounds,
        args=(reference_points, measured_distances),
        popsize=30,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=1e-8,
        maxiter=1000
    )

    spinner("Fine-tuning with L-BFGS-B optimization", delay=0.1, iterations=5)

    result_local = minimize(
        trilateration_objective,
        result_global.x,
        args=(reference_points, measured_distances),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': False, 'ftol': 1e-15, 'gtol': 1e-15, 'maxiter': 15000}
    )

    return result_local.x

def alternative_optimization(reference_points, measured_distances):
    """
    Multi-method optimization approach that tries multiple starting points and methods.
    Returns the best overall solution.
    """
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B']
    best_solution = None
    best_error = float('inf')

    weights = [1/(d**2) for d in measured_distances]
    total_weight = sum(weights)
    initial_lat = sum(lat * w for lat, w in zip(reference_points["lat"], weights)) / total_weight
    initial_lon = sum(lon * w for lon, w in zip(reference_points["lon"], weights)) / total_weight

    for i, method in enumerate(methods):
        spinner(f"Trying optimization method: {method}", delay=0.1, iterations=3)
        try:
            bounds = [(-90, 90), (-180, 180)] if method == 'L-BFGS-B' else None
            result = minimize(
                trilateration_objective,
                [initial_lat, initial_lon],
                args=(reference_points, measured_distances),
                method=method,
                bounds=bounds,
                options={'disp': False, 'maxiter': 10000}
            )

            if result.fun < best_error:
                best_error = result.fun
                best_solution = result.x
        except:
            pass

    antipodal_lat = -initial_lat
    antipodal_lon = initial_lon + 180 if initial_lon < 0 else initial_lon - 180

    for i, method in enumerate(methods):
        spinner(f"Trying antipodal optimization with: {method}", delay=0.1, iterations=2)
        try:
            bounds = [(-90, 90), (-180, 180)] if method == 'L-BFGS-B' else None
            result = minimize(
                trilateration_objective,
                [antipodal_lat, antipodal_lon],
                args=(reference_points, measured_distances),
                method=method,
                bounds=bounds,
                options={'disp': False, 'maxiter': 10000}
            )

            if result.fun < best_error:
                best_error = result.fun
                best_solution = result.x
        except:
            pass

    return best_solution

def geometric_approach(reference_points, measured_distances):
    """
    A more geometrically-focused approach that directly estimates intersection points.
    Works by finding the minimum of the sum of squared distances to the three spheres.
    """
    candidates = grid_search(reference_points, measured_distances, resolution=40)

    best_solution = None
    best_error = float('inf')

    for i, (lat, lon) in enumerate(candidates):
        spinner(f"Optimizing candidate {i+1}/5", delay=0.1, iterations=3)
        result = minimize(
            trilateration_objective,
            [lat, lon],
            args=(reference_points, measured_distances),
            method='L-BFGS-B',
            bounds=[(-90, 90), (-180, 180)],
            options={'disp': False, 'ftol': 1e-15, 'gtol': 1e-15, 'maxiter': 15000}
        )

        if result.fun < best_error:
            best_error = result.fun
            best_solution = result.x

    return best_solution

def plot_results(reference_points, measured_distances, estimated_position):
    """Create a map visualization of the trilateration problem and solution."""
    spinner("Generating visualization", delay=0.1, iterations=10)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    colors = ['red', 'green', 'blue']
    for i in range(len(reference_points["lat"])):
        ax.plot(reference_points["lon"][i], reference_points["lat"][i], 'o',
                color=colors[i], markersize=8, transform=ccrs.PlateCarree(),
                label=f'Reference Point {i+1}')

        circle_points = []
        for angle in range(0, 360, 2):
            try:
                bearing = math.radians(angle)
                dest = geodesic(kilometers=measured_distances[i]/1000).destination(
                    (reference_points["lat"][i], reference_points["lon"][i]), angle)
                circle_points.append((dest.longitude, dest.latitude))
            except:
                pass

        if circle_points:
            circle_points = np.array(circle_points)
            ax.plot(circle_points[:, 0], circle_points[:, 1], '-', color=colors[i],
                    linewidth=1.5, transform=ccrs.PlateCarree(), alpha=0.6)

    ax.plot(estimated_position[1], estimated_position[0], '*', color='black',
            markersize=12, transform=ccrs.PlateCarree(),
            label='Estimated Position')

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="30%", height="30%", loc='lower left',
                      axes_class=plt.Axes)


    sub_ax = fig.add_axes(axins.get_position(), projection=ccrs.PlateCarree())
    axins.remove()
    axins = sub_ax

    # Add features to inset
    axins.add_feature(cfeature.LAND, facecolor='lightgray')
    axins.add_feature(cfeature.OCEAN, facecolor='lightblue')
    axins.add_feature(cfeature.COASTLINE, linewidth=0.5)

    axins.plot(estimated_position[1], estimated_position[0], '*', color='black',
               markersize=10, transform=ccrs.PlateCarree())

    buffer = 2  # degrees
    axins.set_extent([
        estimated_position[1] - buffer,
        estimated_position[1] + buffer,
        estimated_position[0] - buffer,
        estimated_position[0] + buffer
    ], crs=ccrs.PlateCarree())

    gl = axins.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.legend(loc='lower right')
    plt.title('Ultra-Precision Trilateration Map', fontsize=16)

    spinner("Saving visualization image", delay=0.1, iterations=3)
    plt.savefig('trilateration_visualization.png', dpi=300, bbox_inches='tight')
    print(f"{Fore.GREEN}✓ Enhanced visualization saved as 'trilateration_visualization.png'{Style.RESET_ALL}")

def additional_refinement(best_solution, reference_points, measured_distances):
    """
    Additional refinement step using multiple local optimizations with different methods
    and progressively tighter tolerances.
    """
    methods = ['Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B']
    tolerances = [1e-8, 1e-10, 1e-12, 1e-15]

    current_solution = best_solution
    spinner("Starting additional refinement process", delay=0.1, iterations=3)

    for method in methods:
        for tol in tolerances:
            try:
                bounds = [(-90, 90), (-180, 180)] if method == 'L-BFGS-B' else None
                result = minimize(
                    trilateration_objective,
                    current_solution,
                    args=(reference_points, measured_distances),
                    method=method,
                    bounds=bounds,
                    options={'disp': False, 'ftol': tol, 'gtol': tol, 'maxiter': 20000}
                )
                current_solution = result.x
            except:
                pass

    return current_solution

def verify_solution(solution, reference_points, measured_distances):
    """
    Verify the solution by calculating the distances and comparing with measured values.
    Returns residuals and RMS error.
    """
    spinner("Verifying solution accuracy", delay=0.1, iterations=3)

    lat, lon = solution
    residuals = []

    for i in range(len(measured_distances)):
        calculated_distance = vincenty_distance(
            lat, lon,
            reference_points["lat"][i], reference_points["lon"][i]
        )
        residual = calculated_distance - measured_distances[i]
        residuals.append(residual)

    rms_error = np.sqrt(np.mean(np.array(residuals) ** 2))
    return residuals, rms_error

def haversine_distance(lat1, lon1, lat2, lon2):
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ, Δλ = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(a))

def total_distance_error(lat, lon, ref_points, measured_distances):
    errors = []
    for i in range(len(ref_points["lat"])):
        calc_dist = haversine_distance(lat, lon, ref_points["lat"][i], ref_points["lon"][i])
        errors.append(abs(calc_dist - measured_distances[i]))
    return sum(errors), errors

def generate_surrounding_points(base_point, radii):
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    return [(
        geodesic(meters=r).destination(base_point, b).latitude,
        geodesic(meters=r).destination(base_point, b).longitude
    ) for r in radii for b in directions]

def refine_position(base_point, ref_points, measured_distances, max_iter=1000, tolerance=0.00000000000000001):
    current = base_point
    radii = [0.00000000000000001, 0.000000000000001, 0.0000000000001, 0.00000000001, 0.000000001, 0.0000001, 0.00001, 0.001, 0.1, 1, 10, 50, 100, 500, 1000, 2000, 5000, 7500, 10000]
    total_checked = 0
    best_total_error = float('inf')
    improvements = 0
    best_errors = []

    print_header("FINAL TRILATERATION REFINEMENT (Phase 2)")
    print(f"\n{Fore.BLUE}▶ Starting position: {Style.RESET_ALL}{current[0]}, {current[1]}\n")
    for i in range(len(ref_points["lat"])):
        print(f"{Fore.BLUE}▶ Reference point {Style.RESET_ALL}{i+1}: {ref_points['lat'][i]}, {ref_points['lon'][i]}")
    print()
    for i, d in enumerate(measured_distances):
        print(f"{Fore.BLUE}▶ Known distance {Style.RESET_ALL}{i+1}: {d} (meters)")
    print("\n")

    for iteration in range(max_iter):
        surrounding = generate_surrounding_points(current, radii)
        current_total_error, current_errors = total_distance_error(
            current[0], current[1], ref_points, measured_distances
        )

        if iteration == 0:
            best_total_error = current_total_error
            best_errors = current_errors

        best_point = current
        improved = False

        total_checked = (iteration + 1) * 152
        counter_message = f"\rRecursively searching for the right coordinate {iteration + 1}/{total_checked}"
        sys.stdout.write(counter_message + ' ' * 10)  # pad to clear remnants
        sys.stdout.flush()

        for i, pt in enumerate(surrounding):
            total_err, errs = total_distance_error(pt[0], pt[1], ref_points, measured_distances)

            if total_err < best_total_error:
                best_total_error = total_err
                best_errors = errs
                best_point = pt
                improved = True
                improvements += 1

        if improved:
            current = best_point
        else:
            break

        if best_total_error < tolerance:
            break
        
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    print_header("FINAL RESULTS")
    iterations = iteration + 1
    total_checked = iterations * 152
    print(f"\nConverged after {iterations} iterations and {total_checked} tried coordinates with an error of {best_total_error:.4f}m\n")

    latlon = f"Final position: {current[0]}, {current[1]}"
    width = 60
    inner_width = width - 2
    padded = latlon.center(inner_width)

    print("╔" + "═" * inner_width + "╗")
    print("║" + padded + "║")
    print("╚" + "═" * inner_width + "╝")

    print(f"\n")

    for i, err in enumerate(best_errors):
        print(f"Reference point {i+1}: error {err:.4f}m")

    if any(err > 10000 for err in best_errors):
        print(f"\n{Fore.RED}✗ Script failed to calculate distance (failed){Style.RESET_ALL}")
    elif any(5000 < err <= 10000 for err in best_errors):
        print(f"\n{Fore.YELLOW}✗ Precision refinement failed to pinpoint exact location (non-optimal){Style.RESET_ALL}")
    elif any(100 < err <= 5000 for err in best_errors):
        print(f"\n{Fore.YELLOW}✗ Precision refinement failed to pinpoint exact location (sub-optimal){Style.RESET_ALL}")
    elif all(err < 1 for err in best_errors):
        print(f"\n{Fore.GREEN}✓ RMS error is less than a meter, precise result found (optimal){Style.RESET_ALL}")


    return current, best_errors

def main():
    clear_screen()
    print(LOGO)
    time.sleep(1)

    print_header("REFERENCE POINTS INPUT")

    reference_points = {"lat": [], "lon": []}
    measured_distances = []

    for i in range(3):
        print(f"{Fore.BLUE}▶ Reference Point {i+1}:{Style.RESET_ALL}")
        lat = float(input(f"  Enter latitude for point {i+1} (degrees): "))
        lon = float(input(f"  Enter longitude for point {i+1} (degrees): "))
        distance = float(input(f"  Enter distance to unknown point (meters): "))

        reference_points["lat"].append(lat)
        reference_points["lon"].append(lon)
        measured_distances.append(distance)
        print(f"{Fore.GREEN}  ✓ Reference point {i+1} registered{Style.RESET_ALL}\n")

    print_header("OPTIMIZATION PROCESS (Phase 1)")
    print(f"Initiating comprehensive optimization...{Style.RESET_ALL}")
    time.sleep(0.5)

    methods = [
        ("Multi-stage optimization", multi_stage_optimization),
        ("Alternative optimization", alternative_optimization),
        ("Geometric approach", geometric_approach)
    ]

    best_solution = None
    best_error = float('inf')
    best_method = None

    for method_name, method_func in methods:
        try:
            print(f"\nTrying {method_name}...")
            solution = method_func(reference_points, measured_distances)
            error = trilateration_objective(solution, reference_points, measured_distances)

            print(f"  {Fore.BLUE}▶ Solution: {Fore.WHITE}{solution}{Style.RESET_ALL}")
            print(f"  {Fore.BLUE}▶ Error value: {Fore.WHITE}{error}{Style.RESET_ALL}")

            if error < best_error:
                best_error = error
                best_solution = solution
                best_method = method_name
                print(f"  {Fore.GREEN}New best solution found!{Style.RESET_ALL}")
        except Exception as e:
            print(f"  {Fore.RED}✗ Method {method_name} failed: {str(e)}{Style.RESET_ALL}")

    if best_solution is None:
        print(f"{Fore.RED}All optimization methods failed. Please check your input data.{Style.RESET_ALL}")
        return

    print(f"\nPerforming additional refinement...")
    refined_solution = additional_refinement(best_solution, reference_points, measured_distances)
    refined_error = trilateration_objective(refined_solution, reference_points, measured_distances)

    if refined_error < best_error:
        best_solution = refined_solution
        best_error = refined_error
        best_method += " with refinement"
        print(f"  {Fore.GREEN}Refinement improved solution!{Style.RESET_ALL}")

    estimated_lat, estimated_lon = best_solution

    residuals, rms_error = verify_solution(best_solution, reference_points, measured_distances)

    print_header("TRILATERATION RESULTS")
    print_result(f"Best method", best_method, Fore.BLUE)

    print(f"\nEstimated Position:")
    print(f"Latitude:{Style.RESET_ALL}  {Fore.GREEN}{estimated_lat:.8f}°{Style.RESET_ALL}{' ' * (27 - len(f'{estimated_lat:.8f}'))}")
    print(f"Longitude:{Style.RESET_ALL} {Fore.GREEN}{estimated_lon:.8f}°{Style.RESET_ALL}{' ' * (27 - len(f'{estimated_lon:.8f}'))}")


    print(f"\nResiduals (errors in meters):")
    for i, residual in enumerate(residuals):
        print(f"  Point {i+1}: {residual:.2f} meters")

    print(f"\nRMS Error: {rms_error:.2f} meters")

    print_header("VISUALIZATION")
    try:
        plot_results(reference_points, measured_distances, (estimated_lat, estimated_lon))
    except Exception as e:
        print(f"{Fore.RED}Visualization failed: {str(e)}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Note: This does not affect the accuracy of the coordinates.{Style.RESET_ALL}")

    estimated_start = (estimated_lat, estimated_lon)

    final_point, final_errors = refine_position(
        estimated_start, reference_points, measured_distances
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Keyboard interrupt detected, exiting ...{Style.RESET_ALL}")
