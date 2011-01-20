void res(double *A, float *u, float *du, float *beta){
  *du += (*beta)*(*A)*(*u);
}
