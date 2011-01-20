void output(int ncell, int* cell, int type)
{
	int celldim = 0;
	switch(type)
	{
		case 1:
			celldim = 3;
		case 2:
    case 4:
			celldim = 4;
		case 3:
			celldim = 8;
		default:
			return;
	}

	FILE *fp;
  fp = fopen("metis.mesh","w");

	fprintf(fp, "%d %d\n", ncell, type);
	for(int i=0; i<ncell; i++)
	{
		for(int j=0; j<celldim; i++)
		{
			fprintf(fp, "%d", cell[i*celldim + j] + 1);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
}
