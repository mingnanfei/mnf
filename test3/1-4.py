def stretch(wave, factor):
    wave.ts *= factor
    wave.framerate /= factor