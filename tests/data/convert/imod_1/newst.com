$setenv IMOD_OUTPUT_FORMAT MRC
$newstack -StandardInput
AntialiasFilter	-1
InputFile	mka2018-10-01-1.mrc
OutputFile	mka2018-10-01-1_ali.mrc
TransformFile	mka2018-10-01-1.xf
TaperAtFill	1,1
AdjustOrigin
SizeToOutputInXandY	928,958
OffsetsInXandY	0.0,0.0
#DistortionField	.idf
ImagesAreBinned	1.0
BinByFactor	4
#GradientFile	mka2018-10-01-1.maggrad
$if (-e ./savework) ./savework
