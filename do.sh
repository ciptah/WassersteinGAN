CMD="python main.py --dataset cifar10 --dataroot . --cuda \
	--niter 2000 \
	--nz 100 \
	--experiment $EXP \
	--imageSize 32 \
	--clamp_lower -0.009 \
	--clamp_upper 0.009 \
	--Diters 5 \
	--lrD 0.00005 \
	--lrG 0.00005 \
	--nstart 10 \
	--ngf 80 \
	--nef 80 \
	--nconv 64 \
	--ndf 256"

# Switched to DCNN->MLP discriminator
# Switch to 32px images for speed
# Increase clamp size, see what happens
# MLP layer now compares reconstructoins directly

cat $0
$CMD
