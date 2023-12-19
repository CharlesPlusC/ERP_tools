[![HitCount](https://hits.dwyl.com/CharlesPlusC/ERP_tools.svg?style=flat-square&show=unique)](http://hits.dwyl.com/CharlesPlusC/ERP_tools)

<p align="center">
  <img src="misc/UCL-logo-black.jpg" alt="University Logo" width="200"><br/>
  <img src="misc/SGNL_logo_ColouronBlack.jpg" alt="Research Group Logo" width="200">
</p>

<h3 align="center">ERP_tools</h3>

<p align="center">
    Calculate and visualize the radiative flux reaching a given satellite along its trajectory using ERP Tools. This toolset allows users to process satellite data and visualize radiative fluxes in various forms.
  <br />
  <a href="https://github.com/CharlesPlusC/ERP_tools/issues">Report Bug</a>
  ·
  <a href="https://github.com/CharlesPlusC/ERP_tools/pulls">Request Feature</a>
</p>

## Installation

To install the ERP Tools environment and dependencies, follow these steps:

1. Clone the ERP Tools repository:
   ```bash
   git clone https://github.com/CharlesPlusC/ERP_tools.git
   cd ERP_tools

2. Create and Active the conda environment:
   ```bash
   conda env create -f erp_tools_env.yml
   conda activate erp_tools_env
   ```

## Usage

After installing and activating the environment, you can run the toolset by executing the main script:

```bash
python main.py
```

Unfortunately, the only way to download CERES data is through the [ceres.larc.nasa.gov](https://ceres-tool.larc.nasa.gov/ord-tool/jsp/SYN1degEd41Selection.jsp). You will need to create an account and download the data manually. The data should be placed in the `data` directory.

Then in `main.py` replace the path variables with the path to the downloaded data.

This will calculate and plot an animation of the combined longwave and shortwave radiative flux based on the provided satellite trajectory information (currently reads and propagates a TLE but this can easily be modified to accept any ECI ephemeris).

### Combined Flux Animation
This particular animation combines both longwave and shortwave radiative fluxes:
![Combined Flux Animation](output/FOV_sliced_data/combined_flux_animation_nipy.gif)

## Example Animations

The ERP Tools generate several GIFs that visualize the radiative flux. Here are some examples:

Longwave Hourly Flux:
![Longwave Hourly Flux](output/animations/oneweb_lw_hrly_flux.gif)

Longwave Hourly Radiance at TOA:
![Longwave Hourly](output/animations/oneweb_lw_hrly.gif)

Shortwave Hourly Flux:
![Shortwave Hourly Flux](output/animations/oneweb_sw_hrly_flux.gif)

Shortwave Hourly Radiance at TOA:
![Shortwave Hourly](output/animations/oneweb_sw_hrly.gif)

## "Geiger"-style Flux Animation

These plots record the cumulative radiative flux over a given orbit. The flux is plotted onto the satellite's FOV and is integrated over the propagation time. The aim was to get an idea of whether there is any significant time-dependent effects of ERP on the spacecraft. I thought this kind of sensing was similar to a Geiger counter, hence the name.

![OneWeb 12 hour geiger plots](output/FOV_sliced_data/geiger_plots/oneweb_1/cumulative_flux_anim.gif)

## Orekit Custom CERES ERP Force Model

I have implemented a ERP force model for Orekit that uses the CERES data to calculate the radiative flux at a given satellite's position. This force model can be used to propagate a satellite's trajectory and calculate the radiative flux at each point along the trajectory. It is still in need of some refinement but it seems to perform similarly to the Knocke model in Orekit.

HCL Difference between the two models relative to a trajectory with no ERP force model for a Starlink satellite over a 6-hour period:
![Knocke vs CERES Starlink](output/ERP_prop/SL_6hr_Knocke_vs_CERES.jpeg)

## Contributing
Contributions and Issues to ERP Tools are welcome.