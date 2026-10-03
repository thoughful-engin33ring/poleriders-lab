sprbridge.dylib: sprbridge.c farm.c
	# build to a temp file and rename: running games keep their mapped copy intact
	clang -arch x86_64 -dynamiclib -O2 -Wall -DGL_SILENCE_DEPRECATION -framework OpenGL -framework CoreGraphics -framework CoreText -framework CoreFoundation -undefined dynamic_lookup -o $@.tmp $< && mv $@.tmp $@
