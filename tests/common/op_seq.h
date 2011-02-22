//                                                                                       
// headers                                                                               
//                                                                                       
                                                                                         
#include <stdlib.h>                                                                      
#include <stdio.h>                                                                       
#include <string.h>                                                                      
#include <math.h>                                                                        
#include "op_datatypes.h"                                                                
                                                                                         
//                                                                                       
// op_par_loop routine for 2 arguments                                                   
//                                                                                       
                                                                                         
template < class T0, class T1 >
float op_par_loop_2(void (*kernel)( T0*, T1* ),
  op_set set,                                                         
  op_dat<T0> *arg0 ,int idx0 ,op_map *map0 ,op_access acc0 ,          
  op_dat<T1> *arg1 ,int idx1 ,op_map *map1 ,op_access acc1){
	return 0;
}                                                                                               
                                                                                         
//                                                                                       
// op_par_loop routine for 3 arguments                                                   
//                                                                                       
                                                                                         
template < class T0, class T1, class T2 >
float op_par_loop_3(void (*kernel)( T0*, T1*, T2* ),
  op_set set,                                                         
  op_dat<T0> *arg0 ,int idx0 ,op_map *map0 ,op_access acc0 ,          
  op_dat<T1> *arg1 ,int idx1 ,op_map *map1 ,op_access acc1 ,
  op_dat<T2> *arg2 ,int idx2 ,op_map *map2 ,op_access acc2){
	return 0;
}                                                                                        
                                                                                         
//                                                                                       
// op_par_loop routine for 4 arguments                                                   
//                                                                                       
                                                                                         
template < class T0, class T1, class T2, class T3 >
float op_par_loop_4(void (*kernel)( T0*, T1*, T2*, T3* ),
  op_set set,                                                         
  op_dat<T0> *arg0 ,int idx0 ,op_map *map0 ,op_access acc0 ,          
  op_dat<T1> *arg1 ,int idx1 ,op_map *map1 ,op_access acc1 ,
	op_dat<T2> *arg2 ,int idx2 ,op_map *map2 ,op_access acc2 ,
  op_dat<T3> *arg3 ,int idx3 ,op_map *map3 ,op_access acc3){
	return 0;
}                                                                                        
                                                                                         
//                                                                                       
// op_par_loop routine for 5 arguments                                                   
//                                                                                       
                                                                                         
template < class T0, class T1, class T2, class T3, class T4 >
float op_par_loop_5(void (*kernel)( T0*, T1*, T2*, T3*, T4* ),
  op_set set,                                                         
  op_dat<T0> *arg0 ,int idx0 ,op_map *map0 ,op_access acc0 ,          
  op_dat<T1> *arg1 ,int idx1 ,op_map *map1 ,op_access acc1 ,
	op_dat<T2> *arg2 ,int idx2 ,op_map *map2 ,op_access acc2 ,
  op_dat<T3> *arg3 ,int idx3 ,op_map *map3 ,op_access acc3 ,
  op_dat<T4> *arg4 ,int idx4 ,op_map *map4 ,op_access acc4){
	return 0;
}                                                                                        
                                                                                         
//                                                                                       
// op_par_loop routine for 6 arguments                                                   
//                                                                                       
                                                                                         
template < class T0, class T1, class T2, class T3, class T4, class T5 >
float op_par_loop_6(void (*kernel)( T0*, T1*, T2*, T3*, T4*, T5* ),
  op_set set,                                                         
  op_dat<T0> *arg0 ,int idx0 ,op_map *map0 ,op_access acc0 ,          
  op_dat<T1> *arg1 ,int idx1 ,op_map *map1 ,op_access acc1 ,
	op_dat<T2> *arg2 ,int idx2 ,op_map *map2 ,op_access acc2 ,
  op_dat<T3> *arg3 ,int idx3 ,op_map *map3 ,op_access acc3 ,
  op_dat<T4> *arg4 ,int idx4 ,op_map *map4 ,op_access acc4 ,
  op_dat<T5> *arg5 ,int idx5 ,op_map *map5 ,op_access acc5){
	return 0;
}
                                                                                       
//                                                                                       
// op_par_loop routine for 9 arguments                                                   
//                                                                                       
                                                                                         
template < class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8 >
float op_par_loop_9(void (*kernel)( T0*, T1*, T2*, T3*, T4*, T5*, T6*, T7*, T8* ),
  op_set set,                                                         
  op_dat<T0> *arg0 ,int idx0 ,op_map *map0 ,op_access acc0 ,          
  op_dat<T1> *arg1 ,int idx1 ,op_map *map1 ,op_access acc1 ,
	op_dat<T2> *arg2 ,int idx2 ,op_map *map2 ,op_access acc2 ,
  op_dat<T3> *arg3 ,int idx3 ,op_map *map3 ,op_access acc3 ,
  op_dat<T4> *arg4 ,int idx4 ,op_map *map4 ,op_access acc4 ,
  op_dat<T5> *arg5 ,int idx5 ,op_map *map5 ,op_access acc5 ,
  op_dat<T6> *arg6 ,int idx6 ,op_map *map6 ,op_access acc6 ,
  op_dat<T7> *arg7 ,int idx7 ,op_map *map7 ,op_access acc7 , 
  op_dat<T8> *arg8 ,int idx8 ,op_map *map8 ,op_access acc8){
	return 0;
}                                                                         
