$setenv IMOD_OUTPUT_FORMAT MRC
$tilt -StandardInput
InputProjections mba2012-02-01-1_ali.mrc
OutputFile mba2012-02-01-1_full_rec.mrc
IMAGEBINNED 2
TILTFILE mba2012-02-01-1.tlt
THICKNESS 900
RADIAL 0.35 0.035
FalloffIsTrueSigma 1
XAXISTILT 0.0
SCALE 0.0 0.1
PERPENDICULAR
MODE 2
FULLIMAGE 2032 2032
SUBSETSTART 0 0
AdjustOrigin
ActionIfGPUFails 1,2
XTILTFILE mba2012-02-01-1.xtilt
OFFSET 0.0
SHIFT 0.0 0.0
$if (-e ./savework) ./savework
