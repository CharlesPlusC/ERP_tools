import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, Slider, CustomJS
from bokeh.layouts import column
from bokeh.tile_providers import CARTODBPOSITRON_RETINA
from pyproj import Transformer

# Load the data
df = pd.read_csv('/Users/charlesc/Documents/GitHub/ERP_tools/output/DensityInversion/OrbitEnergy/4by4GRACE-FO-A_energy_components_2023-05-04 21:59:42_2023-05-21 00:00:12.csv')


# Coordinate transformation
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
df['x_merc'], df['y_merc'] = transformer.transform(df['lon'].values, df['lat'].values)

# Create ColumnDataSource
source = ColumnDataSource(data=dict(x=[], y=[], color=[]))

# Function to update data based on slider
def update_data(n_points=10):
    source.data = {
        'x': df['x_merc'][:n_points],
        'y': df['y_merc'][:n_points],
        'color': df['HOT_total_diff'][:n_points]
    }

update_data(30)  # Initialize with 30 data points

# Output file
output_file("hot_total_diff_map.html")

# Create figure and add map tile directly
p = figure(x_axis_type="mercator", y_axis_type="mercator", width=800, height=600)
p.add_tile(CARTODBPOSITRON_RETINA)

# Add circle glyphs to the plot
p.circle(x='x', y='y', color='color', source=source, fill_alpha=0.7, size=10)

# Slider for data selection
slider = Slider(start=0, end=len(df), value=30, step=30, title="Number of data points")
slider.js_on_change('value', CustomJS(args=dict(source=source, slider=slider, df=df.to_dict('list')),
                                      code="""
                                          const N = slider.value;
                                          source.data = {
                                              x: df['x_merc'].slice(0, N),
                                              y: df['y_merc'].slice(0, N),
                                              color: df['HOT_total_diff'].slice(0, N)
                                          };
                                      """))

# Layout and show
layout = column(p, slider)
show(layout)