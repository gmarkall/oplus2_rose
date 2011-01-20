void update(float *r, float *du, float *u, float *u_sum, float *u_max){
  *u += *du + 1.0f * (*r);
  *du = 0.0f;
  *u_sum += (*u)*(*u);
  *u_max = MAX(*u_max,*u);
}
