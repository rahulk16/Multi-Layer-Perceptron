#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <string.h>
#define NUMPAT 2216
#define NUMIN  17
#define NUMTEST 998
#define NUMOUT 10


#define rando() ((double)rand()/((double)RAND_MAX+1))



int main() {
    int NUMHID,ch,ch2;

    printf("Enter The No. of Nodes in hidden layer :\n");
    scanf("%d",&NUMHID);
    printf("Enter 1 for epoch stopping criteria, 2 for Delta stopping criteria\n");
    scanf("%d",&ch);
    printf("Enter 1 for Squared Error, 2 Cross-Entropy\n");
    scanf("%d",&ch2);
    int    i, j, k, p, np, op, ranpat[NUMPAT+1], epoch,n,index,count=0,nIH,nHO;

    int    NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT;

    double Input[NUMPAT][NUMIN],Test[NUMTEST][NUMIN],max,converge=0.01,countDelta=0.0,countDelta2=0.0;
    int classnum[NUMPAT],classnum1[NUMPAT];
    double Target[NUMPAT+1][NUMOUT+1]={0};
    double SumH[NUMPAT+1][NUMHID+1], WeightIH[NUMIN+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1],HiddenTest[NUMPAT+1][NUMHID+1];
    double SumO[NUMPAT+1][NUMOUT+1], WeightHO[NUMHID+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
    double DeltaO[NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1];
    double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHO[NUMHID+1][NUMOUT+1];

    double Error, eta = 0.001, alpha = 0.9, smallwt = 0.5;
    nIH=(NUMIN+1)*(NUMHID+1);
    nHO=(NUMHID+1)*(NUMOUT+1);
    i=0;j=0;
    char *line = (char*)malloc(1);
    FILE *file;
    file = fopen("train1.txt", "r");

    for(i=0;i<NUMPAT;++i){
        Input[i][0] = 1;
        for(j=0;j<17;++j){
            if(j==0)
                fscanf(file, "%d",&classnum[i]);
            else{
                fscanf(file, "%lf", &Input[i][j]);
            }
        }
    }



    for(i=0;i<=NUMPAT;i++)
    {
       // printf("%d\n",classnum[i]);
        Target[i][classnum[i]-1]=1;
    }
    for(i=0;i<=NUMPAT;i++)
    {
        for(j=0;j<=NUMOUT;j++)
        {
            //printf("%lf ",Target[i][j]);
        }
       // printf("\n");
    }

    fclose(file);

    for( j = 1 ; j <= NumHidden ; j++ ) {    /* initialize WeightIH and DeltaWeightIH */
        for( i = 0 ; i <= NumInput ; i++ ) {
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    for( k = 1 ; k <= NumOutput ; k ++ ) {    /* initialize WeightHO and DeltaWeightHO */
        for( j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }

    for( epoch = 0 ; epoch <= 1000 ; epoch++) {    /* iterate weight updates */
        for(p = 1 ; p <= NumPattern ; p++ ) {    
            ranpat[p] = p ;
        }
        for( p = 1 ; p <= NumPattern ; p++) {
            np = p + rando() * ( NumPattern + 1 - p ) ;
            op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        }
        Error = 0.0 ;
        for( np = 1 ; np <= NumPattern ; np++ ) {
            /* repeat for all the training patterns */
            p = ranpat[np];
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */
                SumH[p][j] = WeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    SumH[p][j] += Input[p][i] * WeightIH[i][j] ;
                }
                Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
            }
            for( k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
                }
                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;  
                if(ch2==1){
                    Error += 0.5*(Target[p][k]-Output[p][k])*(Target[p][k]-Output[p][k]);
                  
           
                DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ; 
                }  
                if(ch2==2){
                    Error -= ( Target[p][k] * log( Output[p][k] ) + ( 1.0 - Target[p][k] ) * log( 1.0 - Output[p][k] ) ) ;
                    DeltaO[k] = Target[p][k] - Output[p][k];
                }
           
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                SumDOW[j] = 0.0 ;
                for( k = 1 ; k <= NumOutput ; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
                }
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {     /* update weights WeightIH */
                DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
                WeightIH[0][j] += DeltaWeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
                    WeightIH[i][j] += DeltaWeightIH[i][j] ;
                }
            }
            for( k = 1 ; k <= NumOutput ; k ++ ) {    /* update weights WeightHO */
                DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
                WeightHO[0][k] += DeltaWeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
                    WeightHO[j][k] += DeltaWeightHO[j][k] ;
                }
            }
        }
    
       // if( epoch%100 == 0 )
            //fprintf(stdout, "\nEpoch %-5d :   Error = %f", epoch, Error) ;

        if(ch=1){
        if(epoch==100)
        {
            //fprintf(stdout, "\nEpoch %-5d :   Error = %f", epoch, Error) ;
            break;}
        
    }
    else{

        for(i=0;i<=NUMIN;i++)
        {
            for(j=0;j<=NUMHID;j++)
            {
                if(DeltaWeightIH[i][j]<=converge)
                    countDelta++;

            }
        }


        for(i=0;i<=NUMHID;i++)
        {
            for(j=0;j<=NUMOUT;j++)
            {
                if(DeltaWeightHO[i][j]<=converge)
                    countDelta2++;

            }
        }
        if((countDelta>=(0.8*nIH))&&(countDelta2>=(0.8*nHO)))
            break;
    }
}


  //  fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch) ;   /* print network outputs */
    /*for( i = 1 ; i <= NumInput ; i++ ) {
       // fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= NumOutput ; k++ ) {
        //fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for(p=1;p<=NumPattern;p++) {
    //fprintf(stdout, "\n%d\t", p);
        for( i = 1 ; i <= NumInput ; i++ ) {
      //      fprintf(stdout, "%.1f\t", Input[p][i]);
        }
        for( k = 1 ; k <= NumOutput ; k++ ) {
        //    fprintf(stdout, "%.1f\t%.1f\t", Target[p][k], Output[p][k]);
        }
    }*/
    FILE *file1;
    file1 = fopen("test.txt", "r");

    for(i=0;i<NUMTEST;++i){
        Test[i][0] = 1;
        for(j=0;j<17;++j){
            if(j==0)
                fscanf(file1, "%d",&classnum1[i]);
            else{
                fscanf(file1, "%lf", &Test[i][j]);
            }
        }
    }


for(n=0;n<NUMTEST;n++)
{
    for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */
                SumH[n][j] = WeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    SumH[n][j] += Test[n][i] * WeightIH[i][j] ;
                }
                Hidden[n][j] = 1.0/(1.0 + exp(-SumH[n][j])) ;
            }
    for( k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
        SumO[n][k] = WeightHO[0][k] ;
        for( j = 1 ; j <= NumHidden ; j++ ) {
                SumO[n][k] += Hidden[n][j] * WeightHO[j][k] ;
                }
                Output[n][k] = 1.0/(1.0 + exp(-SumO[n][k])) ;

            }
}
printf("---------------------\n");
for(i=0;i<=NUMPAT;i++)
{
    max=Output[i][1];
    for(j=1;j<=NUMOUT;j++)
    {
        if(Output[i][j]>max)
        {
            max=Output[i][j];

           // printf("%lf %lf\n",Output[i][i],max);


            index=j;
        }
        Output[i][j]=0;
       // printf("%lf ",Output[i][j]);
    }
    Output[i][index]=1;
    if(classnum1[i]==index)
        count++;
   // printf("j=%d\n",index);
}

printf("%d",count);
    fprintf(stdout, "\n\nGoodbye!\n\n");
    return 1 ;
}

/*******************************************************************************/
