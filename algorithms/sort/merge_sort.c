//********************************
// C Implementation of merge sort
//*******************************

#include <stdio.h>
#include <stdlib.h>

// Merge the two half into a sorted data.
void merge(int arr[], int l, int m, int r) {
  int i, j, k;
  int n1= m -l +1;
  int n2= r - m;

  /* create temp arrays */
  int L[n1], R[n2];

  /* copy data to temp arrays L[] and R[] */
  for (i = 0; i< n1; i++)
    L[i] = arr[l+i];
  for (j = 0; j<n2; j++)
    R[j] = arr[m+ 1+j];

  /* merge the temp arrays back into arr[l...r] */
  i = 0;
  j = 0;
  k = 0;
  while (i < n1 && j < n2) {
    if (L[i] <= R[i]) {
      arr[k] = L[i];
      i++;
    } else {
      arr[k] = R[j];
      j++;
    }
    k++;
  }

  /* copy the remaining elements of L[], if there are any */
  while (i < n1) {
    arr[k] = L[i];
      i++;
      k++;
  }

  /* copy the remaining elements of L[], if there are any */
  while (j  < n1) {
    arr[k] = R[j];
      j++;
      k++;
  }

  
}

/* l is for left index and r is right index of the sub-array of arr to be sorted */
void mergeSort(int arr[], int l, int r) {
  if (l < r) {
    // same as (l+r)/2, but avoids overflow for large l and h
    int m = l+(r-l)/2;
    mergeSort(arr, l, m);
    mergeSort(arr, m+1, r);

    merge(arr, l, m, r);
  }
}

/* UTILILTY FUNCTIONS */
/* Function to print an array */
void printArray(int A[], int size) {
  int i;
  for (i = 0; i < size; i++)
    printf("%d ", A[i]);
  printf("\n");
}

int main() {
  int arr[] = {12,11,13,5,6,7};
  int arr_size = sizeof(arr)/sizeof(arr[0]);

  printf("Given arrar is \n");
  printArray(arr, arr_size);

  mergeSort(arr, 0, arr_size -1);

  printf("\nSorted arrar is \n");
  printArray(arr, arr_size);
  return 0;
}