from paraview.simple import *
import os

# Path to your time series (change if needed)
series = os.path.expanduser('~/POISSONSOLVER/Results/Oct-4/sweeps/phi_sweep_series.xdmf')
state  = os.path.expanduser('~/POISSONSOLVER/Results/Oct-4/sweeps/center_clip_slice_series.pvsm')

# Load
src = OpenDataFile(series)
AnimationScene().UpdateAnimationUsingDataTimeSteps()

# View
view = GetActiveViewOrCreate('RenderView')
Show(src, view)
disp_src = GetDisplayProperties(src, view)
ColorBy(disp_src, ('POINTS','phi'))

# Rescale color/opacity over *all time steps*
try:
    GetColorTransferFunction('phi').RescaleTransferFunctionToDataRangeOverTime(src, view)
    GetOpacityTransferFunction('phi').RescaleTransferFunctionToDataRangeOverTime(src, view)
except Exception:
    pass

# --- Center Clip (keep half-domain to see inside)
clip = Clip(Input=src)
clip.ClipType = 'Plane'
clip.ClipType.Origin = [0.5, 0.5, 0.5]
clip.ClipType.Normal = [1.0, 0.0, 0.0]   # x=0.5 plane; flip Invert if you want the other half
Show(clip, view)
Hide(src, view)
disp_clip = GetDisplayProperties(clip, view)
ColorBy(disp_clip, ('POINTS','phi'))
disp_clip.Opacity = 0.35  # semi-transparent

# --- Center Slice on the clipped volume (z=0.5)
sl = Slice(Input=clip)
sl.SliceType = 'Plane'
sl.SliceType.Origin = [0.5, 0.5, 0.5]
sl.SliceType.Normal = [0.0, 0.0, 1.0]
Show(sl, view)
disp_sl = GetDisplayProperties(sl, view)
ColorBy(disp_sl, ('POINTS','phi'))

# Rescale LUT/TF over time for the derived objects too
try:
    lut = GetColorTransferFunction('phi')
    pwf = GetOpacityTransferFunction('phi')
    for obj in [clip, sl]:
        lut.RescaleTransferFunctionToDataRangeOverTime(obj, view)
        pwf.RescaleTransferFunctionToDataRangeOverTime(obj, view)
except Exception:
    pass

# Show scalar bar
lut = GetColorTransferFunction('phi')
bar = GetScalarBar(lut, view); bar.Title = 'phi'; bar.ComponentTitle = ''; bar.Visibility = 1

# Save state so you can reopen in GUI
SaveState(state)
print("Saved ParaView state:", state)
