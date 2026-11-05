# Builds the "thres" program and the shared library PhaseScope loads (c/libthreshold.so).
#   make                Linux
#   make mac            macOS        (libthreshold.dylib)
#   make windows        MinGW / MSYS2 (thres.exe and threshold.dll)
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c99

all: thres libthreshold.so

thres: thres_main.c threshold.c threshold.h
	$(CC) $(CFLAGS) -o $@ thres_main.c threshold.c -lm

libthreshold.so: threshold.c threshold.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ threshold.c -lm

mac: thres libthreshold.dylib

libthreshold.dylib: threshold.c threshold.h
	$(CC) $(CFLAGS) -dynamiclib -o $@ threshold.c -lm

windows: thres.exe threshold.dll

thres.exe: thres_main.c threshold.c threshold.h
	$(CC) $(CFLAGS) -o $@ thres_main.c threshold.c -lm

threshold.dll: threshold.c threshold.h
	$(CC) $(CFLAGS) -DTHRESHOLD_BUILD_DLL -shared -o $@ threshold.c -lm

clean:
	rm -f thres thres.exe libthreshold.so libthreshold.dylib threshold.dll

.PHONY: all mac windows clean
