import os, time, numpy as np
import tensorflow as tf
from keras import layers, models

print("TensorFlow version:", tf.__version__)
print("Keras:", tf.keras.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normaliza para [0,1] e adiciona canal (N, 28, 28, 1)
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test = (x_test.astype("float32") / 255.0)[..., None]

print("x_train:", x_train.shape, x_train.dtype)
print("x_test :", x_test.shape, x_test.dtype)
print("y_train:", y_train.shape, " | y_test:", y_test.shape)

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)), 
    layers.Flatten(), 
    layers.Dense(64, activation="relu"), 
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
    
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=2,
    batch_size=256,
    verbose=1
)
    
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"[Keras] Acurácia no teste: {test_acc:.3f}")

os.makedirs("export", exist_ok=True)

keras_path = "export/mnist_teste.keras"
model.save(keras_path)
print("Salvo (Keras v3):", keras_path)

h5_path = "export/mnist_teste.h5"
model.save(h5_path)
print("Salvo (HDF5 legado):", h5_path)

# (A) TFLite float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float32 = converter.convert()
tfl_float_path = "export/mnist_teste_float32.tflite"
open(tfl_float_path, "wb").write(tflite_float32)

# (B) TFLite dynamic range (quantizado)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dynamic = converter.convert()
tfl_dyn_path = "export/mnist_teste_dynamic.tflite"
open(tfl_dyn_path, "wb").write(tflite_dynamic)

try:
    from ai_edge_litert.interpreter import Interpreter
    print("Usando LiteRT (ai-edge-litert).")
except ImportError:
    from tensorflow.lite import Interpreter
    print("Usando tf.lite.Interpreter (fallback).")

try:
    from ai_edge_litert.interpreter import Interpreter
    print("Usando LiteRT (ai-edge-litert).")
except ImportError:
    from tensorflow.lite import Interpreter
    print("Usando tf.lite.Interpreter (fallback).")
 
#Inferência em uma única imagem de teste:
def run_inference(tflite_path, img):
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], img.astype("float32"))
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details["index"])[0]
    pred = int(np.argmax(probs))
    return probs, pred
i = 0
img = x_test[i:i+1]
probs_dyn, pred_dyn = run_inference(tfl_dyn_path, img)
probs_f32, pred_f32 = run_inference(tfl_float_path, img)
probs_keras = model.predict(img, verbose=0)[0]
pred_keras = int(np.argmax(probs_keras))
 
print("🔎 Resultado da previsão")
print("="*40)
print(f"✔️ Rótulo real: {y_test[i]}")
print(f"🤖 Keras       → {pred_keras}")
print(f"⚡ LiteRT f32  → {pred_f32}")
print(f"⚡ LiteRT dyn  → {pred_dyn}")
print("-"*40)
print("Probabilidades (primeiras 5 classes):")
print("Keras   :", np.round(probs_keras[:5], 3))
print("float32 :", np.round(probs_f32[:5], 3))
print("dynamic :", np.round(probs_dyn[:5], 3))

# === Tamanhos de arquivos ===
def size_mb(p):
    return os.path.getsize(p) / (1024 * 1024) 

print("\nTamanhos aproximados:")
print(f" - {keras_keras_path}: {size_mb(keras_keras_path):.2f} MB (Keras v3)")
print(f" - {keras_h5_path}:    {size_mb(keras_h5_path):.2f} MB (HDF5 legado)")
print(f" - {tfl_float_path}:   {size_mb(tfl_float_path):.2f} MB (TFLite float32)")
print(f" - {tfl_dyn_path}:     {size_mb(tfl_dyn_path):.2f} MB (TFLite dynamic)")
 
# === Benchmark de latência (float32 vs dynamic) ===
def create_interpreter(tflite_path):
    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]
    return interp, input_details, output_details

def warmup(interpreter, input_details, output_details, img, runs=5):
    for _ in range(runs):
        interpreter.set_tensor(input_details["index"], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"]) 

def benchmark(interpreter, input_details, output_details, img, runs=100):
    start = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(input_details["index"], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
    end = time.perf_counter()
    total = (end - start) * 1000.0  # ms
    return total / runs
 
# Preparar input único
bench_img = x_test[:1].astype("float32")
 
# float32
interp_f32, in_f32, out_f32 = create_interpreter(tfl_float_path)
warmup(interp_f32, in_f32, out_f32, bench_img, runs=10)
lat_f32_ms = benchmark(interp_f32, in_f32, out_f32, bench_img, runs=100)
 
# dynamic
interp_dyn, in_dyn, out_dyn = create_interpreter(tfl_dyn_path)
warmup(interp_dyn, in_dyn, out_dyn, bench_img, runs=10)
lat_dyn_ms = benchmark(interp_dyn, in_dyn, out_dyn, bench_img, runs=100)
 
print(f"Latência média (100 execuções):")
print(f" - float32 : {lat_f32_ms:.3f} ms")
print(f" - dynamic : {lat_dyn_ms:.3f} ms")
print("Obs.: resultados variam conforme hardware/ambiente.")

