import cupy as cp
dot_kernel_paralell = cp.RawKernel(
    r'''
    extern "C" __global__
    void single_dot_kernel_paralell(float* inputxd, float* lin, float* outputxd, int population_size, int input_size, int output_size)
    {
      int network_id = blockIdx.x;
      int index_in_output = threadIdx.x;

      float value_for_thread = 0.;


      for(int i = 0; i < input_size; i++)
      {
        value_for_thread += inputxd[i] * lin[network_id * input_size * output_size + i*output_size + index_in_output];
      }
      outputxd[network_id * output_size + index_in_output] = value_for_thread;

    }
    ''',
    'single_dot_kernel_paralell'
)


max_pooling_kernel_paralell = cp.RawKernel(
    r'''
    extern "C" __global__
    void max_pooling_kernel_paralell(float* ret_mat, float* temp, int temp_s1, int temp_s2, int temp_s0)
    {
      int network_id = blockIdx.z;
      int i = blockIdx.y;
      int j = blockIdx.x;
      int k = threadIdx.x;
      float maxi = 0.;

      for(int temp_j = j*2; temp_j < min(j*2 + 2, temp_s1) ; temp_j++)
      {
        for(int temp_k = k*2; temp_k < min(k*2 + 2, temp_s2) ; temp_k++)
        {
          float z_tablicy = temp[network_id * temp_s0 *  temp_s1 * temp_s2+ i *  temp_s1 * temp_s2 + temp_j * temp_s2  + temp_k];
          maxi = max(maxi, z_tablicy);
        }
      }

      ret_mat[network_id * gridDim.y * gridDim.x * blockDim.x+ i * gridDim.x * blockDim.x + j * blockDim.x + k] =  maxi;
      
    }
    ''',
    'max_pooling_kernel_paralell'
)

conv_kernel_paralell_many_inputs = cp.RawKernel(
    r'''
    extern "C" __global__
    void conv_kernel_paralell_many_inputs(float* ret_mat, float* temp, float * conv, int temp_s1, int temp_s2, int temp_s3, int conv_s1, int conv_s2, int filtersize)
    {
      int id_network = blockIdx.x; //retmatshape 0
      int i = blockIdx.y; //retmatshape 1 = convshape 1
      int j = blockIdx.z; //retmatshape 2 = tempshape 2
      int k = threadIdx.x; // retmatshape 3 = tempshape 3

      float t = 0.;
      
      for(int temp_i = 0; temp_i < conv_s2; temp_i++)
      {
        for(int temp_j = j; temp_j < j + filtersize; temp_j++)
        {
          for(int temp_k = k; temp_k < k + filtersize; temp_k++)
          {
            t += temp[id_network * temp_s1 * temp_s2 * temp_s3 + temp_i * temp_s2 * temp_s3 + temp_j * temp_s3 + temp_k] * 
            conv[id_network * conv_s1 * conv_s2 * filtersize * filtersize+ i * conv_s2 * filtersize * filtersize + temp_i * filtersize * filtersize + (temp_j - j) * filtersize + temp_k - k];
          }
        }
      }

      ret_mat[id_network * gridDim.z * gridDim.y * blockDim.x+ i * gridDim.z * blockDim.x + j * blockDim.x + k] = t;
      
    }
    ''',
    'conv_kernel_paralell_many_inputs'
)

dot_kernel_paralell_many_inputs = cp.RawKernel(
    r'''
    extern "C" __global__
    void dot_kernel_paralell_many_inputs(float* input, float* lin, float* output, int population_size, int input_size, int single_input_size, int output_size)
    {
      int network_id = blockIdx.x;
      int index_in_output = threadIdx.x;

      float value_for_thread = 0.;


      int index_in_input = network_id*single_input_size;
      for(int i = index_in_input; i < index_in_input + single_input_size; i++)
      {
        value_for_thread += input[i] * lin[network_id * single_input_size * output_size + (i - index_in_input)*output_size + index_in_output];
      }
      output[network_id * output_size + index_in_output] = value_for_thread;

    }
    ''',
    'dot_kernel_paralell_many_inputs'
)



conv_kernel_paralell = cp.RawKernel(
    r'''
    extern "C" __global__
    void conv_kernel_paralell(float* ret_mat, float* temp, float * conv, int temp_s1, int temp_s2, int temp_s3, int conv_s1, int conv_s2, int filtersize)
    {
      int id_network = blockIdx.x; //retmatshape 0
      int i = blockIdx.y; //retmatshape 1 = convshape 1
      int j = blockIdx.z; //retmatshape 2 = tempshape 2
      int k = threadIdx.x; // retmatshape 3 = tempshape 3

      float t = 0.;
      
      for(int temp_i = 0; temp_i < conv_s2; temp_i++) //input filter number
      {
        for(int temp_j = j; temp_j < j + filtersize; temp_j++)
        {
          for(int temp_k = k; temp_k < k + filtersize; temp_k++)
          {
            t += temp[temp_i * temp_s2 * temp_s3 +  temp_j * temp_s3 + temp_k] * 
            conv[id_network * conv_s1 * conv_s2 * filtersize * filtersize+ i * conv_s2 * filtersize * filtersize + temp_i * filtersize * filtersize + (temp_j - j) * filtersize + temp_k - k];
          }
        }
      }

      ret_mat[id_network * gridDim.z * gridDim.y * blockDim.x+ i * gridDim.z * blockDim.x + j * blockDim.x + k] = t;
      
    }
    ''',
    'conv_kernel_paralell'
)

def max_pooling_cuda_paralell(temp):
    ret_mat = cp.zeros((temp.shape[0], temp.shape[1], int(cp.ceil(temp.shape[2]/2)), int(cp.ceil(temp.shape[3]/2))), dtype = cp.float32)
    block_size = (ret_mat.shape[3],1)
    grid_size =  (ret_mat.shape[2], ret_mat.shape[1], ret_mat.shape[0])
    max_pooling_kernel_paralell(grid_size, block_size, (ret_mat, temp, temp.shape[2], temp.shape[3], temp.shape[1]))
    return ret_mat

def convolve_cuda_paralell_many_inputs(temp, conv):
    #int temp_s1, int temp_s2, int temp_s3, int conv_s1, int conv_s2, int filtersize
    ret_mat = cp.zeros((temp.shape[0] , conv.shape[1], temp.shape[2] - 2, temp.shape[3] - 2), dtype = cp.float32)
    block_size = (ret_mat.shape[3], 1)
    grid_size =  (ret_mat.shape[0], ret_mat.shape[1], ret_mat.shape[2])
    conv_kernel_paralell_many_inputs(grid_size, block_size, (ret_mat, temp, conv,temp.shape[1], temp.shape[2], temp.shape[3], conv.shape[1], conv.shape[2], 3))
    return ret_mat


def dot_cuda_paralell_many_inputs(input, lin):
  ret_mat = cp.zeros((lin.shape[0], lin.shape[2]), dtype=cp.float32)
  block_size = (lin.shape[2], 1)
  grid_size = (lin.shape[0], 1)

  input_size = cp.int32(input.shape[0])
  single_input_size = cp.int32(input.shape[0]//lin.shape[0])
  output_size = cp.int32(lin.shape[2])
  population_size = cp.int32(lin.shape[0])
  dot_kernel_paralell_many_inputs(grid_size, block_size, (input, lin, ret_mat, population_size, input_size, single_input_size, output_size))
  return ret_mat 


def dot_cuda_paralell(single_input, lin):
  ret_mat = cp.zeros((lin.shape[0], lin.shape[2]), dtype=cp.float32)
  block_size = (lin.shape[2], 1)
  grid_size = (lin.shape[0], 1)

  input_size = cp.int32(single_input.shape[0])
  output_size = cp.int32(lin.shape[2])
  population_size = cp.int32(lin.shape[0])
  dot_kernel_paralell(grid_size, block_size, (single_input, lin, ret_mat, population_size, input_size, output_size))
  return ret_mat 


def convolve_cuda_paralell(temp, conv):
    #int temp_s1, int temp_s2, int temp_s3, int conv_s1, int conv_s2, int filtersize
    ret_mat = cp.zeros((conv.shape[0] , conv.shape[1], temp.shape[1] - 2, temp.shape[2] - 2), dtype = cp.float32)
    block_size = (ret_mat.shape[3], 1)
    grid_size =  (ret_mat.shape[0], ret_mat.shape[1], ret_mat.shape[2])
    conv_kernel_paralell(grid_size, block_size, (ret_mat, temp, conv, temp.shape[0], temp.shape[1], temp.shape[2], conv.shape[1], conv.shape[2], 3))
    return ret_mat
