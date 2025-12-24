import onnx
from onnx import helper
from onnx import TensorProto
import os

def create_dummy_fas_model():
    # Define input and output
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 128, 128])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1]) # Single score

    # Create a node that produces a constant output
    # We want it to output something that indicates "Real" (score > 0.5)
    
    # Constant node
    output_tensor = helper.make_tensor(
        name='const_tensor',
        data_type=TensorProto.FLOAT,
        dims=[1, 1],
        vals=[0.9] # Mock value: Real
    )
    
    node_def = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['output'],
        value=output_tensor
    )

    # Create the graph
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [input_info],
        [output_info],
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    
    # Save
    if not os.path.exists('models'):
        os.makedirs('models')
        
    onnx.save(model_def, 'models/fas.onnx')
    print("Created dummy FAS model at models/fas.onnx")

if __name__ == "__main__":
    create_dummy_fas_model()
