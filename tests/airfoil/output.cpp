//
// standard headers
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

float gam, gm1, cfl, eps, mach, alpha;

#include "op_seq.h"
#include "input.h"

using namespace std;


enum eCellType
{
	CELL_TYPE_TRIANGLE = 1,
	CELL_TYPE_TETRAHEDRAL = 2,
	CELL_TYPE_HEXAHEDRAL = 3,
  CELL_TYPE_QUADRILATERAL = 4
};

void output(int ncell, int* cell, int type)
{
	const char* name = "metis.mesh";
	int celldim = 0;
	switch(type)
	{
		case 1:
			celldim = 3;
			break;
		case 2:
    case 4:
			celldim = 4;
			break;
		case 3:
			celldim = 8;
			break;
		default:
			return;
	}

	FILE *fp;
  fp = fopen(name,"w");

	fprintf(fp, "%d %d\n", ncell, type);
	for(int i=0; i<ncell; i++)
	{
		for(int j=0; j<celldim; j++)
		{
			fprintf(fp, "%d ", cell[i*celldim + j] + 1);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);

	// Export bash script
	int bsize = 64;
	int gridsize = (ncell - 1) / bsize + 1;

	fp = fopen("part.sh","w");

	fprintf(fp, "#!/bin/bash\n");
	fprintf(fp, "../metis-4.0/partdmesh ./%s %d", name, gridsize);

	fclose(fp);
}


int main(int argc, char **argv)
{
#define maxnode  9900
#define maxcell (9702+1)
#define maxedge 19502

  int    ecell[2*maxedge], boun[maxedge], edge[2*maxedge], cell[4*maxcell] = {0};
  float x[2*maxnode],q[4*maxcell],qold[4*maxcell],adt[maxcell],res[4*maxcell];

  int    nnode,ncell,nedge, niter;
  float rms;

  // read in grid and flow data

  input(maxnode,maxcell,maxedge,nnode,ncell,nedge,x,q,cell,edge,ecell,boun);
	

	output(ncell, cell, CELL_TYPE_QUADRILATERAL);
}
