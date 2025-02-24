import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from marzip_extractor import MarzipExtractor

def angular_diff(value, mean):
    """Returns the minimal difference between two angles (°), result range: [-180, 180)."""
    return (value - mean + 180) % 360 - 180

class TargetShipTimeseriesAnalyzer(MarzipExtractor):
    def analyze_timeseries(self, marzip_file, output_dir):
        """Analyze the target ship time-series data from a .marzip file and generate plots."""
        # Extract data from the marzip file (using parent's run())
        self.run(marzip_file)
        
        # Collect IDs of targets whose "movement" is empty
        targets = self.simulation_result.get("trafficSituation", {}).get("targetShips", [])
        valid_ids = set()
        for target in targets:
            if not target.get("movement"):
                tid = target.get("static", {}).get("id")
                if tid is not None:
                    valid_ids.add(tid)
        print("Valid target IDs:", valid_ids)
        
        # Group timeseries data by IDs in valid_ids
        groups = {}
        for entry in self.timeseries_dataset:
            entry_id = entry.get("id")
            if entry_id not in valid_ids:
                continue
            if entry_id not in groups:
                groups[entry_id] = {"sog": [], "cog": [], "lat": [], "lon": []}
            
            # Parse data safely
            if "sog" in entry:
                try:
                    groups[entry_id]["sog"].append(float(entry["sog"]))
                except Exception as e:
                    print(f"SOG parse error (file: {marzip_file}, id: {entry_id}): {e}")
            if "cog" in entry:
                try:
                    groups[entry_id]["cog"].append(float(entry["cog"]))
                except Exception as e:
                    print(f"COG parse error (file: {marzip_file}, id: {entry_id}): {e}")
            if "lat" in entry:
                try:
                    groups[entry_id]["lat"].append(float(entry["lat"]))
                except Exception as e:
                    print(f"Lat parse error (file: {marzip_file}, id: {entry_id}): {e}")
            if "lon" in entry:
                try:
                    groups[entry_id]["lon"].append(float(entry["lon"]))
                except Exception as e:
                    print(f"Lon parse error (file: {marzip_file}, id: {entry_id}): {e}")
        
        base_name = os.path.splitext(os.path.basename(marzip_file))[0]
        
        for target_id, data in sorted(groups.items(), key=lambda x: x[0]):
            print(f"\nFile: {marzip_file} / Target ID: {target_id}")
            first_sog = data["sog"][0] if data["sog"] else None
            first_cog = data["cog"][0] if data["cog"] else None
            print("First SOG:", first_sog)
            print("First COG:", first_cog)
            
            # SOG histogram with Gaussian fit
            values = data.get("sog", [])
            if values:
                values = np.array(values)
                plt.figure(figsize=(8,6))
                plt.hist(values, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')
                
                mu, std = norm.fit(values)
                if std > 1e-12:
                    # Draw normal PDF
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, mu, std)
                    plt.plot(x, p, 'r', linewidth=2)
                    
                    sigma_colors = ['black', 'blue', 'green', 'orange', 'red']
                    for n in range(1, 6):
                        plt.axvline(mu + n*std, color=sigma_colors[n-1], linestyle='--', linewidth=1)
                        plt.axvline(mu - n*std, color=sigma_colors[n-1], linestyle='--', linewidth=1)
                    title = f"SOG Histogram & Gaussian Fit\nμ = {mu:.2f}, σ = {std:.2f}"
                else:
                    # Degenerate case: all values are the same (std ~ 0)
                    title = (f"SOG Histogram - Degenerate (All ~ {mu:.2f})\n"
                             f"(Std. dev. = 0, no Gaussian curve)")
                
                plt.title(title)
                plt.xlabel("SOG")
                plt.ylabel("Density")
                output_file = os.path.join(output_dir, f"{base_name}_{target_id}_sog_histogram_gaussian.png")
                plt.savefig(output_file)
                plt.close()
                print(f"Saved SOG histogram with Gaussian fit to {output_file}")
            else:
                print(f"No SOG data (file: {marzip_file}, id: {target_id})")
            
            # COG histogram: offset relative to the first COG
            values = data.get("cog", [])
            if values:
                values = np.mod(np.array(values), 360)
                initial_cog = values[0]  # first COG
                adjusted_cog = np.array([initial_cog + angular_diff(v, initial_cog) for v in values])
                
                sigma_data = np.std(adjusted_cog - initial_cog)
                
                plt.figure(figsize=(8,6))
                if sigma_data > 1e-12:
                    bins = np.linspace(initial_cog - 5*sigma_data, initial_cog + 5*sigma_data, 31)
                else:
                    bins = 31
                
                plt.hist(adjusted_cog, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black')
                
                if sigma_data > 1e-12:
                    x_range = np.linspace(initial_cog - 5*sigma_data, initial_cog + 5*sigma_data, 100)
                    p = norm.pdf(x_range, loc=initial_cog, scale=sigma_data)
                    plt.plot(x_range, p, 'r', linewidth=2)
                    sigma_colors = ['black', 'blue', 'green', 'orange', 'red']
                    for n in range(1, 6):
                        plt.axvline(initial_cog + n*sigma_data, color=sigma_colors[n-1], linestyle='--', linewidth=1)
                        plt.axvline(initial_cog - n*sigma_data, color=sigma_colors[n-1], linestyle='--', linewidth=1)
                    title = f"COG Offset Histogram & Gaussian Fit\nCenter = {initial_cog:.2f}°, σ = {sigma_data:.2f}°"
                else:
                    title = (f"COG Offset Histogram - Degenerate\n"
                             f"(Center = {initial_cog:.2f}°, Std. dev. = 0, no Gaussian curve)")
                
                plt.title(title)
                plt.xlabel("COG (°)")
                plt.ylabel("Density")
                output_file = os.path.join(output_dir, f"{base_name}_{target_id}_cog_offset_histogram.png")
                plt.savefig(output_file)
                plt.close()
                print(f"Saved COG offset histogram with Gaussian fit to {output_file}")
            else:
                print(f"No COG data (file: {marzip_file}, id: {target_id})")
            
            # Lat-Lon plot (NM units)
            lat_values = data.get("lat", [])
            lon_values = data.get("lon", [])
            if lat_values and lon_values:
                ref_lat = lat_values[0]
                ref_lon = lon_values[0]
                
                # Convert from degrees to NM
                lat_nm = [(lat - ref_lat) * 60 for lat in lat_values]
                lon_nm = [(lon - ref_lon) * 60 * np.cos(np.radians(ref_lat)) for lon in lon_values]
                
                plt.figure(figsize=(8,6))
                plt.scatter(lon_nm, lat_nm, s=10, color="purple", label="Data Points")
                plt.scatter([0], [0], s=50, color="red", marker="*", label="Initial Point")
                
                if len(lon_nm) >= 2:
                    # Simple linear fit
                    coeff = np.polyfit(lon_nm, lat_nm, 1)
                    slope, intercept = coeff
                    x_fit = np.linspace(min(lon_nm), max(lon_nm), 100)
                    y_fit = slope * x_fit + intercept
                    plt.plot(x_fit, y_fit, 'g--', linewidth=2,
                             label=f"Linear Fit: y={slope:.2f}x + {intercept:.2f}")
                    
                    # Residual analysis
                    lon_nm_array = np.array(lon_nm)
                    fitted = slope * lon_nm_array + intercept
                    residuals = np.array(lat_nm) - fitted
                    
                    res_std = np.std(residuals)    # in NM
                    position_std_m = res_std * 1852  # convert NM to meters
                    
                    plt.text(
                        0.05, 0.95,
                        f"Position Std. Dev.: {position_std_m:.2f} m", 
                        transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.7)
                    )
                    
                    # Plot histogram of position residuals in meters
                    plt.figure(figsize=(8,6))
                    residuals_m = residuals * 1852
                    plt.hist(residuals_m, bins=20, density=True, alpha=0.6, color='lightgreen', edgecolor='black')
                    
                    if res_std > 1e-12:
                        mu, std = norm.fit(residuals_m)
                        x_min, x_max = plt.xlim()
                        x_range = np.linspace(x_min, x_max, 100)
                        p = norm.pdf(x_range, mu, std)
                        plt.plot(x_range, p, 'r', linewidth=2)
                        title_res = (f"Position Residual Histogram (Target {target_id})\n"
                                     f"Mean: {mu:.2f} m, Std. Dev.: {std:.2f} m")
                    else:
                        title_res = (f"Position Residual Histogram (Target {target_id}) - Degenerate\n"
                                     f"(Std. dev. = 0, no Gaussian curve)")
                    
                    plt.title(title_res)
                    plt.xlabel("Residual (m)")
                    plt.ylabel("Density")
                    
                    output_file_res = os.path.join(
                        output_dir, f"{base_name}_{target_id}_position_variance_histogram.png"
                    )
                    plt.savefig(output_file_res)
                    plt.close()
                    print(f"Saved position variance histogram to {output_file_res}")
                else:
                    print(f"Not enough data points for linear fit (file: {marzip_file}, id: {target_id})")
                
                plt.title(f"Lat-Lon Trajectory (NM) - Target {target_id}")
                plt.xlabel("Longitude (NM)")
                plt.ylabel("Latitude (NM)")
                plt.legend()
                plt.axis('equal')
                
                output_file = os.path.join(output_dir, f"{base_name}_{target_id}_lat_lon_dashed_NM.png")
                plt.savefig(output_file)
                plt.close()
                print(f"Saved lat-lon trajectory plot (NM) to {output_file}")
            else:
                print(f"No lat/lon data (file: {marzip_file}, id: {target_id})")

    def get_sampled_marzip_files_by_folder(self, base_dir):
        """Pick one random .marzip file from each subfolder in base_dir."""
        sampled_files = []
        for root, dirs, files in os.walk(base_dir):
            marzip_files = [os.path.join(root, f) for f in files if f.endswith('.marzip')]
            if marzip_files:
                sampled_files.append(random.choice(marzip_files))
        print(f"Sampled one file per folder: total {len(sampled_files)} files selected.")
        return sampled_files

def main():
    # Data directory
    base_data_dir = "/media/avikus/One Touch/HinasControlSilsCA/CA_v0.1.4_data/SiLS_sever2/Random_Testing/20250221_1/output"
    
    # Output base directory: replicate the same subfolder structure under this
    output_base_dir = "timeseries/SiLS_sever2/Random_Testing/20250221_1"
    
    analyzer = TargetShipTimeseriesAnalyzer()
    sampled_files = analyzer.get_sampled_marzip_files_by_folder(base_data_dir)
    
    for file_path in sampled_files:
        try:
            # Get relative path from base_data_dir
            rel_path = os.path.relpath(file_path, base_data_dir)
            # Extract folder from relative path
            rel_dir = os.path.dirname(rel_path)
            # Create output path accordingly
            output_dir = os.path.join(output_base_dir, rel_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Perform analysis and save plots
            analyzer.analyze_timeseries(file_path, output_dir)
        except Exception as e:
            print(f"Error processing file ({file_path}): {e}")

if __name__ == "__main__":
    main()
