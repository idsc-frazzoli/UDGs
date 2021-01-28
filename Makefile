tag=gokart-mpc
out-docker=out-docker
forcespro-path=""


solver_dir = src/gokartmpcc/MPCCgokart
generated=\
	$(PWD)/$(solver_dir)/include \
	$(PWD)/$(solver_dir)/lib_target/libMPCCgokart.so \
	$(PWD)/$(solver_dir)/MPCCgokart_interface.c \
	$(PWD)/$(solver_dir)/MPCCgokart_model.h \
	$(PWD)/$(solver_dir)/MPCCgokart_model.c


build:
	docker build -t $(tag) .

build-no-cache:
	docker build --no-cache -t $(tag) .

run:
	mkdir -p $(out-docker)
	docker run -it --user $$(id -u) \
		-v $(PWD)/$(out-docker):/$(out-docker) $(tag) \
		python srcpy/gokart-mpc/model.py
#-v $(forcespro-path):/forces/pro:ro \

black:
	black -l 110 --target-version py38 src


out="out-target"
export-for-target:
	mkdir -p $(out)
	cp -r $(generated) $(out)
