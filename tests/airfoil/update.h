void update(float *qold, float *q, float *res, float *adt, float *rms){
  float del=0.0f, adti=0.0f;

  if (*adt>0.0f) adti = 1.0f/(*adt);

  for (int n=0; n<4; n++) {
    del    = adti*res[n];
    q[n]   = qold[n] - del;
    res[n] = 0.0f;
    *rms  += del*del*(air_const.a1[0]-air_const.a1[1]);
  }
}
