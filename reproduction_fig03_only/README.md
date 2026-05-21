# Figure 03 Only Reproduction

This folder is a standalone extraction for Figure 03 from:

Chen et al., "Joint Radar-Communication Transmission: A Generalized Pareto Optimization Framework," IEEE TSP, 2021.

Only the Figure 03 code path is included. The code keeps the original module style:

- `fParam.py`: system parameters, unit conversions, fixed Figure 03 channels
- `fChannel.py`: Figure 03 channel extraction
- `fCalculations.py`: steering vector, beampattern, DPSL, SINR
- `fAlgorithms.py`: Pareto SDP optimization
- `fPlot.py`: Figure 03 plotting
- `main.py`: procedural Figure 03 experiment script

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python .\main.py
```

The figure is saved to:

```text
figures/paper_fig3.png
```

Optional solver settings:

```powershell
python .\main.py --solver CLARABEL --max-iter 3000
python .\main.py --solver SCS --max-iter 5000
```
