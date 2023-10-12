from flask import Flask, render_template_string
import csv
import numpy as np
import hdbscan
import folium
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster
from shapely.geometry import Polygon, Point
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime

app = Flask(__name__)
matplotlib.use('Agg')

@app.route('/map', methods=['GET'])
def map():
    coords = extract_coords_from_csv()
    coords_array = np.array(coords)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    cluster_labels = clusterer.fit_predict(coords_array)

    m = folium.Map(location=[np.mean(coords_array[:, 0]), np.mean(coords_array[:, 1])], zoom_start=5)
    
    # Add the coords_array to FastMarkerCluster
    m.add_child(FastMarkerCluster(coords_array.tolist()))

    # Northeast Boundary(Blue)
    midwest_coords = [
        (45.249213, -83.636001),
        (45.600575, -84.720699),
        (44.331405, -88.058230),
        (46.357814, -84.627376),
        (47.360169, -87.986799),
        (46.646022, -90.869279),
        (46.687648, -92.264425),
        (48.043352, -90.141377),
        (48.567862, -93.295619),#
        (48.967667, -95.176033),
        (48.962474, -104.019891),
        (41.072816, -103.981820),
        (41.095677, -102.101407),
        (37.112146, -101.924989),
        (37.134369, -94.756198),
        (36.679735, -94.484272),
        (36.720194, -89.423415),
        (37.219314, -89.196243),
        (37.984222, -87.946800),
        (38.073693, -86.129428),
        (39.255088, -84.827881),
        (39.125735, -84.816542),
        (38.430374, -82.493317),
        (39.619278, -80.847379),
        (41.893739, -80.587494),

    ]
    
    folium.Polygon(midwest_coords, color="blue", fill=True, fill_color="blue").add_to(m)

    northeast_coords = [
        (47.292478, -68.271950),
        (44.960087, -74.852614),
        (42.864680, -78.744014),
        (41.884015, -80.396008),
        (39.944778, -80.337008),# end of pensdylavia
        (39.217229, -76.944521),# Maryland
        (39.240080, -74.732029),
        (40.979709, -72.303986),
        (41.682642, -70.431727),
        (43.245413, -70.554902),
        (44.065276, -69.101438),
        (44.734107, -67.106005),
    ]
    folium.Polygon(northeast_coords, color="gray", fill=True, fill_color="gray").add_to(m)
    
    # Southeast Boundary (Yellow)
    southeast_coords = [
        (36.719164, -76.016032), # Bottom right of Virginia
        (39.143463, -77.627587), #Upper virginia
        (39.626207, -79.78763), #Upper West Virginia
        (39.324883, -81.375134), # Cover West Virginia
        (39, -83), # Up to Kentucky
        (37.5, -88.5), # Entire top of Kentucky
        (36, -89.5), # Top of Tennessee
        (36.3, -94), # Up a bit to cover top of Arkansas
        (33, -94), # Entire left of Arkansas
        (30, -93.5), # Bottom of Louisiana
        (29.288423, -90.300964),#Continued
        (30.425101, -89.027741), #Progressing towards FLorida
        (30.101659, -85.769789), #Continued
        (29.777155, -83.522925), #start of florida
        (27.907554, -82.811418), #Progressing downwards of Florida
        (26.408256, -81.837778), #Continued
        (25.228449, -80.602003), #Bottom of Florida
        (26.977020, -80.077734), #Upwards of Flordia
        (30.308446, -81.405010), #Continued
        (31.596512, -81.242142), #Upwards back to Virginia
        (33.049221, -79.39630), #Continued
        (33.049221, -79.396300), #Continued
    ]
    # Southeast Boundary (Yellow)
    folium.Polygon(southeast_coords, color="yellow", fill=True, fill_color="yellow").add_to(m)

    # Southwest Boundary (Green)
    southwest_coords = [
        (29, -94.5), # Bottom right of Texas
        (31, -94.5), # Up covering right side of Texas
        (33.59156, -94.089), # Top right corner of Texas
        (33.627115, -94.474516), # Left of Texas
        (37.057965, -94.648852), # top of Oklahoma

        (36.998979, -102.002197), # Oklahoma-Colorado border where it meets New Mexico
        (36.993076, -103.002884), # Oklahoma-New Mexico border
        (36.993076, -104.002884), # Moving west along the top of New Mexico
        (36.994623, -106.005859), # Further west along the top of New Mexico
        (36.999080, -108.003906), # Approaching Arizona

        (37.026932, -109.063929), # Start of Arizona
        (36.985003, -111.037882), # Moving west in Arizona near Monument Valley
        (36.971491, -113.079102), # More west in Arizona 

        (37.009939, -114.053797), # top left corner of Arizona
        (32.547344, -114.763766), # Bottom left of Arizona
        (31.475286, -111.097057), #Bottom right of Arizona
        (31.504686, -108.299802), #Bottom of New Mexico
        (31.813186, -108.206060), #Continued
        (31.802796, -106.555636), #Continued
        (28.988734, -103.235439), #Bottom of Texas
        (29.745317, -102.719973), #Continued
        (29.367728, -100.956536), #Souther texas border
        (27.048929, -99.410138), #Continued
        (25.980797, -97.294014), #Bottom of Texas
    ]
    # Southwest Boundary (Green)
    folium.Polygon(southwest_coords, color="green", fill=True, fill_color="green").add_to(m)

    # West Boundary (Red)
    west_coords = [
        (32.555271, -117.092383), # Bottom left of California
        (32.865057, -114.567600), # Bottom right of California
        (35.039586, -114.676299), # Cover bottom of Nevada start
        (36.057220, -114.699904), #Cover bottom of Nevada middle
        (36.115044, -114.008228), #Cover bottom of Nevada end
        (37.028018, -114.033727), #Bottom left Utah
        (37.020226, -109.054601), # Cover bottom sides of Utah 
        (37.020226, -102.068127), # Cover bottom Colorado
        (41.012128, -102.068127), # Right side of Colorado
        (41.012359, -104.039624), # Some top of Colorado
        (45.011651, -104.056027), # Entire right side of Wyoming
        (48.985175, -104.058622), # Montana
        (48.983457, -123.282896), #Top left washington
        (48.266763, -124.547926), #Washington
        (45.846007, -123.915411), #Oregon
        (43.227410, -124.367207), #Continued
        (40.438692, -124.20550), #Califronia left
        (37.329760, -122.389569), #Continued
        (34.541322, -120.413487), #Continued
        (33.644599, -118.098426), #Continued
        (32.555271, -117.092383), #Bottom left corner
    ]
    # West Boundary (Red)
    folium.Polygon(west_coords, color="red", fill=True, fill_color="red").add_to(m)

    southeast_boundary = Polygon(southeast_coords)
    southwest_boundary = Polygon(southwest_coords)
    west_boundary = Polygon(west_coords)
    northeast_boundary = Polygon(northeast_coords)
    midwest_boundary = Polygon(midwest_coords)

    southeast_count = 0
    southwest_count = 0
    west_count = 0
    northeast_count = 0  # Initialize the Northeast count
    midwest_count = 0  # Initialize the Midwest count

    for coord in coords:
        point = Point(coord)
        if point.within(southeast_boundary):
            southeast_count += 1
        elif point.within(southwest_boundary):
            southwest_count += 1
        elif point.within(west_boundary):
            west_count += 1
        elif point.within(northeast_boundary):
            northeast_count += 1  # Count wildfires in the Northeast
        elif point.within(midwest_boundary):
            midwest_count += 1  # Count wildfires in the Midwest

    print(f"Wildfires in Southeast Region: {southeast_count}")
    print(f"Wildfires in Southwest Region: {southwest_count}")
    print(f"Wildfires in West Region: {west_count}")
    print(f"Wildfires in Northeast Region: {northeast_count}")
    print(f"Wildfires in Midwest Region: {midwest_count}")

    total_counts = [
        southeast_count, southwest_count, 
        west_count, northeast_count, midwest_count
    ]
    max_count = max(total_counts)
    
    # Calculate the total number of wildfires
    total_wildfires = southeast_count + southwest_count + northeast_count + midwest_count + west_count

    # Compute susceptibility for each region
    southeast_susceptibility = southeast_count / total_wildfires
    southwest_susceptibility = southwest_count / total_wildfires
    northeast_susceptibility = northeast_count / total_wildfires
    midwest_susceptibility = midwest_count / total_wildfires
    west_susceptibility = west_count/total_wildfires

    print("Southeast Susceptibility:", southeast_susceptibility)
    print("Southwest Susceptibility:", southwest_susceptibility)
    print("Northeast Susceptibility:", northeast_susceptibility)
    print("Midwest Susceptibility:", midwest_susceptibility)
    print("West Susceptibility:", west_susceptibility)

# Data
    total_counts = [southeast_count, southwest_count, west_count, northeast_count, midwest_count]
    regions = ['Southeast', 'Southwest', 'West', 'Northeast', 'Midwest']
    colors = ['#FFD700', '#32CD32', '#FF4500', '#A9A9A9', '#4682B4']

    # Create a larger figure for clarity
    plt.figure(figsize=(12, 8))

    # Bar chart with color differentiation
    bars = plt.bar(regions, total_counts, color=colors, alpha=0.7)  # Added alpha for a softer color effect

    # Annotation of bar values on top of each bar
    for bar, count in zip(bars, total_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), 
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Title and labels
    plt.title("Number of Wildfires by Region", fontsize=20, fontweight='bold', fontfamily='sans-serif', pad=20)
    plt.xlabel("Regions", fontsize=14, fontweight='bold', fontfamily='sans-serif', labelpad=15)
    plt.ylabel("Number of Wildfires", fontsize=14, fontweight='bold', fontfamily='sans-serif', labelpad=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Adjust y-axis range as per your requirement
    plt.ylim(0, 80000)

    plt.tight_layout()

    # Save and show the bar chart
    plt.savefig("wildfires_by_region_counts.png", dpi=300, bbox_inches="tight")
    plt.show()

# Data
    susceptibilities = [southeast_susceptibility, southwest_susceptibility, west_susceptibility, northeast_susceptibility, midwest_susceptibility]
    regions = ['Southeast', 'Southwest', 'West', 'Northeast', 'Midwest']
    colors = ['yellow', 'green', 'red', 'gray', 'blue']

    # Create a larger figure for clarity
    plt.figure(figsize=(10, 7))

    # Plot the donut chart
    wedges, texts = plt.pie(susceptibilities, colors=colors, startangle=140, wedgeprops=dict(width=0.3, edgecolor='w'))

    # Create a function to format the labels with percentages and region name
    def func(pct, allvals, region):
        return "{}\n{:.1f}%".format(region, pct)

    # Annotation and placing the labels
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        plt.annotate(func(susceptibilities[i]/sum(susceptibilities)*100, susceptibilities, regions[i]), xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), horizontalalignment=horizontalalignment, **kw)

    # Title
    plt.title("Wildfire Susceptibility by Region", fontsize=16, fontweight='bold', fontfamily='sans-serif')

    plt.tight_layout()  # Adjust layout for better appearance

    # Save and show the donut chart
    plt.savefig("wildfire_susceptibility_donut.png", dpi=300)  # Save with higher resolution suitable for print
    plt.show()

    map_html = m._repr_html_()

    return render_template_string(f"<html><body>{map_html}</body></html>")


def extract_coords_from_csv():
    with open("wildfires_data.csv", 'r') as file:
        reader = csv.DictReader(file)
        coords = []
        for row in reader:
            latitude_value = row.get('LATITUDE')
            longitude_value = row.get('LONGITUDE')

            if latitude_value and longitude_value and is_float(latitude_value) and is_float(longitude_value):
                coords.append([float(latitude_value), float(longitude_value)])
    return coords

@app.route('/date', methods=['GET'])
def extract_discovery_times_from_csv():
    with open("wildfires_data.csv", 'r') as file:
        reader = csv.DictReader(file)
        discovery_dates = []
        for row in reader:
            year_value = row.get('FIRE_YEAR')  # Assuming 'FIRE_YEAR' is the column name for the year
            julian_date_value = row.get('DISCOVERY_DOY')  # Assuming 'DISCOVERY_DOY' is the Julian date
            
            print(f"Year: {year_value}, Julian Date: {julian_date_value}")  # Print the extracted values

            if year_value and julian_date_value and year_value.isdigit() and julian_date_value.isdigit():
                gregorian_date = julian_to_gregorian(int(year_value), int(julian_date_value))
                print(f"Converted Gregorian Date: {gregorian_date}")  # Print the converted date
                discovery_dates.append(gregorian_date)
        print("All processed discovery dates:", discovery_dates)  # Print all the discovery dates at the end
    return discovery_dates


def julian_to_gregorian(year, julian_day):
    # Given a year and a Julian day (1-based day of the year), return a Gregorian date.
    return datetime.datetime(year, 1, 1) + datetime.timedelta(julian_day - 1)


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    app.run(debug=True)
