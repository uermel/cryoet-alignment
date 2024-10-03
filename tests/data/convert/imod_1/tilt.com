$setenv IMOD_OUTPUT_FORMAT MRC
$tilt -StandardInput
InputProjections mka2018-10-01-1_ali.mrc
OutputFile mka2018-10-01-1_full_rec.mrc
IMAGEBINNED 4
TILTFILE mka2018-10-01-1.tlt
XTILTFILE mka2018-10-01-1.xtilt
THICKNESS 2000
RADIAL 0.35 0.035
FalloffIsTrueSigma 1
XAXISTILT 0.0
LOG 0.0
SCALE 0.0 330.0
PERPENDICULAR
MODE 1
FULLIMAGE 3708 3832
SUBSETSTART -2 0
AdjustOrigin
OFFSET 0.0
SHIFT 0.0 0.0
ActionIfGPUFails 1,2
$if (-e ./savework) ./savework
