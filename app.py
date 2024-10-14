import streamlit as st
import pandas as pd
import numpy as np

# Funciones definidas anteriormente
horizontal_kernel = np.array([[-1, -1, -1],
                              [ 2,  2,  2],
                              [-1, -1, -1]])

vertical_kernel = np.array([[-1,  2, -1],
                            [-1,  2, -1],
                            [-1,  2, -1]])

def apply_convolution (matrix, kernel_type='horizontal'):
    if kernel_type == 'horizontal':
        kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    elif kernel_type == 'vertical':
        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    else:
        raise ValueError("El tipo de kernel debe ser 'horizontal' o 'vertical'")
    
    rows, cols = matrix.shape
    output = np.zeros((rows - 2, cols - 2))
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = matrix[i-1:i+2, j-1:j+2]
            output[i-1, j-1] = np.sum(region * kernel)
    
    return output

def add_padding(matrix, padding_size):
    padded_matrix = np.pad(matrix, pad_width=padding_size, mode='constant', constant_values=0)
    return padded_matrix

def apply_stride_convolution(matrix, kernel_type='horizontal', stride=2):
    convolved_matrix = apply_convolution(matrix, kernel_type)
    result = convolved_matrix[::stride, ::stride]
    return result

def generate_feature_maps(matrix, n):
    feature_maps = []
    for i in range(n):
        if i % 2 == 0:
            feature_map = apply_convolution(matrix, 'horizontal')
        else:
            feature_map = apply_convolution(matrix, 'vertical')
        feature_maps.append(feature_map)
    return np.stack(feature_maps, axis=-1)

def max_pooling(matrix, stride=2):
    output_shape = (matrix.shape[0] // stride, matrix.shape[1] // stride)
    pooled_matrix = np.zeros(output_shape)
    
    for i in range(0, matrix.shape[0] - 1, stride):
        for j in range(0, matrix.shape[1] - 1, stride):
            pooled_matrix[i//stride, j//stride] = np.max(matrix[i:i+2, j:j+2])
    
    return pooled_matrix

# Interfaz de usuario con Streamlit
st.title('Transformaciones de Matrices con Streamlit')

# Subir archivo
uploaded_file = st.file_uploader("Sube el archivo 'pixeles.xlsx'", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Leer el archivo Excel
        data = pd.read_excel(uploaded_file, sheet_name='Hoja1')
        matrix = data.to_numpy()

        # Mostrar la matriz original
        st.write("Matriz Original:")
        st.dataframe(data)

        # Seleccionar la función a aplicar
        option = st.selectbox(
            'Seleccione la operación que desea realizar',
            ('Convolución', 'Padding', 'Stride', 'Stacking', 'Max Pooling'))

        if option == 'Convolución':
            kernel_type = st.radio('Seleccione el tipo de kernel', ('horizontal', 'vertical'))
            if st.button('Calcular'):
                result = apply_convolution(matrix, kernel_type)
                st.write("Resultado de la Convolución:")
                st.dataframe(pd.DataFrame(result))

        elif option == 'Padding':
            padding_size = st.number_input('Ingrese el tamaño de padding', min_value=1, value=1)
            if st.button('Calcular'):
                result = add_padding(matrix, padding_size)
                st.write("Resultado del Padding:")
                st.dataframe(pd.DataFrame(result))

        elif option == 'Stride':
            kernel_type = st.radio('Seleccione el tipo de kernel', ('horizontal', 'vertical'))
            stride = st.number_input('Ingrese el valor del stride', min_value=1, value=2)
            if st.button('Calcular'):
                result = apply_stride_convolution(matrix, kernel_type, stride)
                st.write("Resultado de la Convolución con Stride:")
                st.dataframe(pd.DataFrame(result))

        elif option == 'Stacking':
            num_maps = st.number_input('Ingrese la cantidad de mapas a generar', min_value=1, value=2)
            if st.button('Calcular'):
                result = generate_feature_maps(matrix, num_maps)
                st.write(f"Resultado del Stacking con {num_maps} mapas:")
                for i in range(result.shape[-1]):
                    st.write(f"Mapa de características {i+1}:")
                    st.dataframe(pd.DataFrame(result[:,:,i]))

        elif option == 'Max Pooling':
            stride = st.number_input('Ingrese el valor del stride', min_value=1, value=2)
            if st.button('Calcular'):
                result = max_pooling(matrix, stride)
                st.write("Resultado del Max Pooling:")
                st.dataframe(pd.DataFrame(result))
    
    except Exception as e:
        st.write(f"Error al procesar el archivo: {e}")
else:
    st.write("Esperando a que se cargue el archivo.")

