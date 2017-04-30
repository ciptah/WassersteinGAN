CMD="python main.py --dataset cifar10 --dataroot . --cuda \
	--niter 2000 \
	--nz 100 \
	--experiment $EXP \
	--imageSize 32 \
	--clamp_lower -0.015 \
	--clamp_upper 0.015 \
	--Diters 8 \
	--adam \
	--nstart 15 \
	--ndf 128"

# Switched to DCNN->MLP discriminator
# Switch to 32px images for speed
# Increase clamp size, see what happens
# MLP layer now compares reconstructoins directly

cat $0
$CMD
