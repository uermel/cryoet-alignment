$setenv IMOD_OUTPUT_FORMAT MRC
$newstack -StandardInput
AntialiasFilter	-1
InputFile	mba2012-02-01-1.mrc
OutputFile	mba2012-02-01-1_ali.mrc
TransformFile	mba2012-02-01-1.xf
TaperAtFill	1,1
AdjustOrigin
OffsetsInXandY	0.0,0.0
#DistortionField	.idf
ImagesAreBinned	1.0
BinByFactor	2
#GradientFile	.maggrad
$if (-e ./savework) ./savework
