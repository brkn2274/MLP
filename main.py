import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sympy.physics.control.control_plots import matplotlib

matplotlib.use('TkAgg')
import time

# Veri Yükleme ve Ön İşleme
df = pd.read_csv(r"C:\Users\cbark\Downloads\BankNote_Authentication.csv")  # r ile kaçış karakteri hatası önlenir.

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy().reshape(-1, 1)

# Veri normalizasyonu eklendi
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Aktivasyon Fonksiyonları
def sigmoid(Z):
    # Taşmaları önlemek için sınırlandırma
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def tanh(Z):
    return np.tanh(Z)


# ----------------- 2-KATMANLI MLP (1 GİZLİ KATMAN) -----------------

# 2-Katmanlı MLP için Parametrelerin Başlatılması
def initialize_parameters_2layer(n_x, n_h, n_y=1):
    np.random.seed(42)
    # He başlatma yöntemi
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2. / n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * np.sqrt(2. / n_h)
    b2 = np.zeros((n_y, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# 2-Katmanlı MLP için İleri Yayılım
def forward_propagation_2layer(X, parameters, activation="tanh"):
    if activation == "tanh":
        act_func = tanh
    elif activation == "relu":
        act_func = relu
    else:
        raise ValueError("Geçersiz aktivasyon fonksiyonu!")

    W1, b1, W2, b2 = parameters.values()

    # X'in doğru formatta olduğundan emin olalım
    if X.ndim == 1:
        X = X.reshape(1, -1)

    Z1 = np.dot(W1, X.T) + b1
    A1 = act_func(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


# 2-Katmanlı MLP için Geri Yayılım
def backpropagation_2layer(X, Y, cache, parameters, activation="tanh"):
    m = X.shape[0]

    W1, W2 = parameters["W1"], parameters["W2"]
    Z1, A1, Z2, A2 = cache["Z1"], cache["A1"], cache["Z2"], cache["A2"]

    # Çıkış katmanı hatası
    dZ2 = A2 - Y.T

    # Geri yayılım - çıkış katmanı
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # Gizli katmana hata yayılımı
    if activation == "tanh":
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * (1 - np.power(A1, 2))
    elif activation == "relu":
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * (Z1 > 0)
    else:
        raise ValueError("Geçersiz aktivasyon fonksiyonu!")

    # Geri yayılım - giriş katmanı
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# 2-Katmanlı MLP Tahmin Fonksiyonu
def predict_2layer(X, parameters, activation="tanh"):
    A2, _ = forward_propagation_2layer(X, parameters, activation)
    return (A2 > 0.5).astype(int).T


# ----------------- 3-KATMANLI MLP (2 GİZLİ KATMAN) -----------------

# 3-Katmanlı MLP için Parametrelerin Başlatılması
def initialize_parameters_3layer(n_x, n_h1, n_h2, n_y=1):
    np.random.seed(42)
    # He başlatma yöntemi
    W1 = np.random.randn(n_h1, n_x) * np.sqrt(2. / n_x)
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * np.sqrt(2. / n_h1)
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) * np.sqrt(2. / n_h2)
    b3 = np.zeros((n_y, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


# 3-Katmanlı MLP için İleri Yayılım
def forward_propagation_3layer(X, parameters, activation="tanh"):
    if activation == "tanh":
        act_func = tanh
    elif activation == "relu":
        act_func = relu
    else:
        raise ValueError("Geçersiz aktivasyon fonksiyonu!")

    W1, b1, W2, b2, W3, b3 = parameters.values()

    # X'in doğru formatta olduğundan emin olalım
    if X.ndim == 1:
        X = X.reshape(1, -1)

    Z1 = np.dot(W1, X.T) + b1
    A1 = act_func(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = act_func(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    return A3, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}


# Kayıp Fonksiyonu (Loss Function) - güvenli hesaplama eklendi
def compute_cost(A_final, Y):
    m = Y.shape[0]
    # Numerik stabilitesi için epsilon ekleyelim
    epsilon = 1e-8
    A_final = np.clip(A_final, epsilon, 1 - epsilon)
    cost = -np.sum(Y.T * np.log(A_final) + (1 - Y.T) * np.log(1 - A_final)) / m
    return float(np.squeeze(cost))


# 3-Katmanlı MLP için Geri Yayılım
def backpropagation_3layer(X, Y, cache, parameters, activation="tanh"):
    m = X.shape[0]

    W1, W2, W3 = parameters["W1"], parameters["W2"], parameters["W3"]
    Z1, A1, Z2, A2, Z3, A3 = cache["Z1"], cache["A1"], cache["Z2"], cache["A2"], cache["Z3"], cache["A3"]

    # Çıkış katmanı hatası
    dZ3 = A3 - Y.T

    # Geri yayılım - 2. gizli katman
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    # 2. gizli katmana hata yayılımı
    if activation == "tanh":
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = dA2 * (1 - np.power(A2, 2))
    elif activation == "relu":
        dA2 = np.dot(W3.T, dZ3)
        dZ2 = dA2 * (Z2 > 0)
    else:
        raise ValueError("Geçersiz aktivasyon fonksiyonu!")

    # Geri yayılım - 1. gizli katman
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # 1. gizli katmana hata yayılımı
    if activation == "tanh":
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * (1 - np.power(A1, 2))
    elif activation == "relu":
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * (Z1 > 0)
    else:
        raise ValueError("Geçersiz aktivasyon fonksiyonu!")

    # Geri yayılım - giriş katmanı
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}


# 3-Katmanlı MLP Tahmin Fonksiyonu
def predict_3layer(X, parameters, activation="tanh"):
    A3, _ = forward_propagation_3layer(X, parameters, activation)
    return (A3 > 0.5).astype(int).T


# Parametre Güncelleme - Momentum eklendi
def update_parameters(parameters, grads, v=None, learning_rate=0.01, beta=0.9):
    if v is None:
        v = {}
        for key in parameters.keys():
            v["v" + key] = np.zeros_like(parameters[key])

    for key in parameters.keys():
        v["v" + key] = beta * v["v" + key] + (1 - beta) * grads["d" + key]
        parameters[key] -= learning_rate * v["v" + key]

    return parameters, v


# 2-Katmanlı MLP için Model Eğitimi
def train_mlp_2layer(X, Y, n_x, n_h, n_y, n_epochs=1000, batch_size=32, learning_rate_init=0.1, activation="tanh", verbose=True):
    parameters = initialize_parameters_2layer(n_x, n_h, n_y)
    v = None
    m = X.shape[0]
    costs = []

    # Öğrenme hızı azaltma
    learning_rate = learning_rate_init

    for epoch in range(n_epochs):
        # Mini-batch SGD
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :]

        num_complete_minibatches = m // batch_size

        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * batch_size:(k + 1) * batch_size, :]
            mini_batch_Y = shuffled_Y[k * batch_size:(k + 1) * batch_size, :]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        if m % batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * batch_size:, :]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * batch_size:, :]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        total_cost = 0
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch

            A2, cache = forward_propagation_2layer(mini_batch_X, parameters, activation)
            cost = compute_cost(A2, mini_batch_Y)
            total_cost += cost * len(mini_batch_X) / m

            grads = backpropagation_2layer(mini_batch_X, mini_batch_Y, cache, parameters, activation)
            parameters, v = update_parameters(parameters, grads, v, learning_rate)

        # Her 100 adımda bir maliyet yazdırma
        if epoch % 100 == 0 and verbose:
            print(f"Epoch {epoch}, Cost: {total_cost:.4f}, Learning rate: {learning_rate:.4f}")
            costs.append(total_cost)

        # Öğrenme hızını azaltma (annealing)
        if epoch % 500 == 0 and epoch > 0:
            learning_rate = learning_rate * 0.5

    return parameters, costs


# 3-Katmanlı MLP için Model Eğitimi
def train_mlp_3layer(X, Y, n_x, n_h1, n_h2, n_y, n_epochs=1000, batch_size=32, learning_rate_init=0.1, activation="tanh", verbose=True):
    parameters = initialize_parameters_3layer(n_x, n_h1, n_h2, n_y)
    v = None
    m = X.shape[0]
    costs = []

    # Öğrenme hızı azaltma
    learning_rate = learning_rate_init

    for epoch in range(n_epochs):
        # Mini-batch SGD
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :]

        # Önceki kodun devamı...

        num_complete_minibatches = m // batch_size

        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * batch_size:(k + 1) * batch_size, :]
            mini_batch_Y = shuffled_Y[k * batch_size:(k + 1) * batch_size, :]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        if m % batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * batch_size:, :]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * batch_size:, :]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        total_cost = 0
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch

            A3, cache = forward_propagation_3layer(mini_batch_X, parameters, activation)
            cost = compute_cost(A3, mini_batch_Y)
            total_cost += cost * len(mini_batch_X) / m

            grads = backpropagation_3layer(mini_batch_X, mini_batch_Y, cache, parameters, activation)
            parameters, v = update_parameters(parameters, grads, v, learning_rate)

        # Her 100 adımda bir maliyet yazdırma
        if epoch % 100 == 0 and verbose:
            print(f"Epoch {epoch}, Cost: {total_cost:.4f}, Learning rate: {learning_rate:.4f}")
            costs.append(total_cost)

        # Öğrenme hızını azaltma (annealing)
        if epoch % 500 == 0 and epoch > 0:
            learning_rate = learning_rate * 0.5

    return parameters, costs


# Modellerin değerlendirilmesi için metrik fonksiyonu
def evaluate_model(y_true, y_pred, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{model_name} Performans Metrikleri:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\nDetaylı Sınıflandırma Raporu:")
    print(classification_report(y_true, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


# Hyperparameter tuning için deneysel fonksiyon
def run_experiments(X_train, y_train, X_test, y_test, n_x):
    # Deneylerin sonuçlarını saklamak için
    results = {
        "2layer": {},
        "3layer": {}
    }

    # Hyperparameter aralıkları
    neurons_range = [3, 5, 8, 10]
    iterations_range = [100, 300, 500, 1000]
    activations = ["tanh", "relu"]

    best_accuracy_2layer = 0
    best_config_2layer = None
    best_params_2layer = None

    best_accuracy_3layer = 0
    best_config_3layer = None
    best_params_3layer = None

    for n_h in neurons_range:
        for n_iter in iterations_range:
            for act in activations:
                print(f"\n{'=' * 50}")
                print(f"2-Katmanlı MLP: n_h={n_h}, iterasyon={n_iter}, aktivasyon={act}")

                # 2-Katmanlı MLP eğitimi
                start_time = time.time()
                parameters_2layer, _ = train_mlp_2layer(
                    X_train, y_train, n_x, n_h, 1,
                    n_epochs=n_iter,
                    batch_size=32,
                    learning_rate_init=0.1,
                    activation=act,
                    verbose=False
                )
                train_time = time.time() - start_time

                # Test verisi üzerinde değerlendirme
                y_pred_2layer = predict_2layer(X_test, parameters_2layer, activation=act)
                accuracy = accuracy_score(y_test, y_pred_2layer)
                print(f"Test Accuracy: {accuracy:.4f}, Training Time: {train_time:.2f} seconds")

                # En iyi modeli saklama
                if accuracy > best_accuracy_2layer:
                    best_accuracy_2layer = accuracy
                    best_config_2layer = (n_h, n_iter, act)
                    best_params_2layer = parameters_2layer

                # Sonucu sakla
                config_key = f"n_h={n_h}_iter={n_iter}_act={act}"
                results["2layer"][config_key] = {
                    "accuracy": accuracy,
                    "time": train_time
                }

                # 3-Katmanlı MLP için ikinci gizli katmanın nöron sayısı
                for n_h2 in neurons_range:
                    print(f"\n{'=' * 50}")
                    print(f"3-Katmanlı MLP: n_h1={n_h}, n_h2={n_h2}, iterasyon={n_iter}, aktivasyon={act}")

                    # 3-Katmanlı MLP eğitimi
                    start_time = time.time()
                    parameters_3layer, _ = train_mlp_3layer(
                        X_train, y_train, n_x, n_h, n_h2, 1,
                        n_epochs=n_iter,
                        batch_size=32,
                        learning_rate_init=0.1,
                        activation=act,
                        verbose=False
                    )
                    train_time = time.time() - start_time

                    # Test verisi üzerinde değerlendirme
                    y_pred_3layer = predict_3layer(X_test, parameters_3layer, activation=act)
                    accuracy = accuracy_score(y_test, y_pred_3layer)
                    print(f"Test Accuracy: {accuracy:.4f}, Training Time: {train_time:.2f} seconds")

                    # En iyi modeli saklama
                    if accuracy > best_accuracy_3layer:
                        best_accuracy_3layer = accuracy
                        best_config_3layer = (n_h, n_h2, n_iter, act)
                        best_params_3layer = parameters_3layer

                    # Sonucu sakla
                    config_key = f"n_h1={n_h}_n_h2={n_h2}_iter={n_iter}_act={act}"
                    results["3layer"][config_key] = {
                        "accuracy": accuracy,
                        "time": train_time
                    }

    print("\n\n" + "=" * 70)
    print("DENEY SONUÇLARI")
    print("=" * 70)
    print(f"\nEn iyi 2-Katmanlı MLP: {best_config_2layer}, Accuracy: {best_accuracy_2layer:.4f}")
    print(f"En iyi 3-Katmanlı MLP: {best_config_3layer}, Accuracy: {best_accuracy_3layer:.4f}")

    # En iyi modeli seçme
    final_model_type = "2layer" if best_accuracy_2layer >= best_accuracy_3layer else "3layer"
    final_params = best_params_2layer if best_accuracy_2layer >= best_accuracy_3layer else best_params_3layer
    final_activation = best_config_2layer[2] if best_accuracy_2layer >= best_accuracy_3layer else best_config_3layer[3]

    print(f"\nSeçilen model: {final_model_type}")

    # Seçilen modelin tahminleri
    if final_model_type == "2layer":
        y_pred = predict_2layer(X_test, final_params, activation=final_activation)
    else:
        y_pred = predict_3layer(X_test, final_params, activation=final_activation)

    # Final değerlendirme
    print("\nFinal Model Performansı:")
    evaluate_model(y_test, y_pred, model_name=f"En iyi {final_model_type}")

    # Sonuçların görselleştirilmesi
    visualize_results(results)

    return results, final_model_type, final_params, final_activation


# Sonuçların görselleştirilmesi
def visualize_results(results):
    # 2-katmanlı model sonuçları
    accuracies_2layer = [value["accuracy"] for value in results["2layer"].values()]
    times_2layer = [value["time"] for value in results["2layer"].values()]

    # 3-katmanlı model sonuçları
    accuracies_3layer = [value["accuracy"] for value in results["3layer"].values()]
    times_3layer = [value["time"] for value in results["3layer"].values()]

    # Doğruluk dağılımı
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(accuracies_2layer, alpha=0.5, label="2-Katmanlı MLP")
    plt.hist(accuracies_3layer, alpha=0.5, label="3-Katmanlı MLP")
    plt.xlabel("Doğruluk (Accuracy)")
    plt.ylabel("Frekans")
    plt.title("Model Doğruluk Dağılımı")
    plt.legend()

    # Eğitim süresi dağılımı
    plt.subplot(1, 2, 2)
    plt.hist(times_2layer, alpha=0.5, label="2-Katmanlı MLP")
    plt.hist(times_3layer, alpha=0.5, label="3-Katmanlı MLP")
    plt.xlabel("Eğitim Süresi (saniye)")
    plt.ylabel("Frekans")
    plt.title("Eğitim Süresi Dağılımı")
    plt.legend()

    plt.tight_layout()
    plt.savefig("mlp_experiment_results.png")
    plt.show()


# Scikit-learn MLP ile karşılaştırma
def compare_with_sklearn(X_train, y_train, X_test, y_test, best_config):
    # En iyi konfigürasyondan parametreleri çıkarma
    if len(best_config) == 3:  # 2-layer için
        n_h, n_iter, activation = best_config
        hidden_layer_sizes = (n_h,)
    else:  # 3-layer için
        n_h1, n_h2, n_iter, activation = best_config
        hidden_layer_sizes = (n_h1, n_h2)

    # Scikit-learn MLP sınıflandırıcısı
    print("\nScikit-learn MLP ile karşılaştırma yapılıyor...")
    start_time = time.time()
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu' if activation == 'relu' else 'tanh',
        solver='sgd',
        batch_size=32,
        learning_rate_init=0.1,
        max_iter=n_iter,
        random_state=42
    )
    clf.fit(X_train, y_train.ravel())
    train_time = time.time() - start_time

    y_pred_sklearn = clf.predict(X_test)

    print(f"Scikit-learn MLP eğitim süresi: {train_time:.2f} saniye")
    evaluate_model(y_test, y_pred_sklearn, model_name="Scikit-learn MLP")

    return clf


# Ana uygulama
if __name__ == "__main__":
    # Toplam özellik sayısı (giriş)
    n_x = X_train.shape[1]

    # Modellerin deneysel karşılaştırması
    print("MLP modelleri üzerinde arama uzayı değerlendirmesi başlatılıyor...")
    results, selected_model, best_params, best_activation = run_experiments(X_train, y_train, X_test, y_test, n_x)

    # En iyi konfigürasyon
    if selected_model == "2layer":
        n_h, n_iter, act = [key for key, val in results[selected_model].items()
                            if val["accuracy"] == max([v["accuracy"] for v in results[selected_model].values()])][
            0].split("_")
        best_config = (int(n_h.split("=")[1]), int(n_iter.split("=")[1]), act.split("=")[1])
    else:
        n_h1, n_h2, n_iter, act = [key for key, val in results[selected_model].items()
                                   if
                                   val["accuracy"] == max([v["accuracy"] for v in results[selected_model].values()])][
            0].split("_")
        best_config = (int(n_h1.split("=")[1]), int(n_h2.split("=")[1]), int(n_iter.split("=")[1]), act.split("=")[1])

    # En iyi yapılandırmayı scikit-learn ile karşılaştırma
    sklearn_model = compare_with_sklearn(X_train, y_train, X_test, y_test, best_config)

    print("\nÖdev sonucu:")
    if selected_model == "2layer":
        print(f"En iyi model: 2-Katmanlı MLP (1 gizli katman)")
        print(f"Gizli katman nöron sayısı: {best_config[0]}")
    else:
        print(f"En iyi model: 3-Katmanlı MLP (2 gizli katman)")
        print(f"1. gizli katman nöron sayısı: {best_config[0]}")
        print(f"2. gizli katman nöron sayısı: {best_config[1]}")

    print(f"İterasyon sayısı: {best_config[-2]}")
    print(f"Aktivasyon fonksiyonu: {best_config[-1]}")

# Test setiyle final performans
if selected_model == "2layer":
    print("\nTest seti üzerinde final değerlendirme:")
    y_pred_final = predict_2layer(X_test, best_params, activation=best_activation)
    final_metrics = evaluate_model(y_test, y_pred_final, model_name="Final 2-Katmanlı MLP")
else:
    print("\nTest seti üzerinde final değerlendirme:")
    y_pred_final = predict_3layer(X_test, best_params, activation=best_activation)
    final_metrics = evaluate_model(y_test, y_pred_final, model_name="Final 3-Katmanlı MLP")

# PyTorch ile kıyaslama
print("\nPyTorch implementasyonu ile karşılaştırma yapılıyor...")


# PyTorch MLP sınıfı
class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='tanh'):
        super(PyTorchMLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Katmanları oluştur
        layers = []
        prev_size = input_size

        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            else:  # ReLU
                layers.append(nn.ReLU())
            prev_size = h_size

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# PyTorch modeli oluştur ve eğit
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)

# Model konfigürasyonu
if selected_model == "2layer":
    hidden_sizes = [best_config[0]]
else:
    hidden_sizes = [best_config[0], best_config[1]]

start_time = time.time()
torch_model = PyTorchMLP(n_x, hidden_sizes, 1, activation=best_activation)
criterion = nn.BCELoss()
optimizer = optim.SGD(torch_model.parameters(), lr=0.1)

# Eğitim döngüsü
n_epochs = best_config[-2]
batch_size = 32
num_batches = len(X_train) // batch_size

for epoch in range(n_epochs):
    # Mini-batch eğitimi
    permutation = torch.randperm(len(X_train_tensor))

    for i in range(0, len(X_train_tensor), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        # İleri yayılım
        outputs = torch_model(batch_x)
        loss = criterion(outputs, batch_y)

        # Geri yayılım
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Her 100 adımda bir maliyet yazdırma
    if epoch % 100 == 0:
        with torch.no_grad():
            total_outputs = torch_model(X_train_tensor)
            total_loss = criterion(total_outputs, y_train_tensor)
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

    # Öğrenme hızını azaltma
    if epoch % 500 == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5

train_time = time.time() - start_time
print(f"PyTorch MLP eğitim süresi: {train_time:.2f} saniye")

# Test verisi üzerinde değerlendirme
with torch.no_grad():
    y_pred_torch = torch_model(X_test_tensor)
    y_pred_torch = (y_pred_torch > 0.5).numpy().astype(int)

evaluate_model(y_test, y_pred_torch, model_name="PyTorch MLP")

# Sonuçların karşılaştırmalı grafiği
models = ["Manuel MLP", "Scikit-learn MLP", "PyTorch MLP"]
accuracies = [
    accuracy_score(y_test, y_pred_final),
    accuracy_score(y_test, y),
    accuracy_score(y_test, y_pred_torch)
]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.ylabel('Doğruluk (Accuracy)')
plt.title('Farklı MLP Implementasyonlarının Karşılaştırması')
plt.ylim(0.9, 1.0)  # Daha iyi görselleştirme için y eksenini sınırla
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig("mlp_comparison.png")
plt.show()

print("\nÖdev Sonucu:")
print(f"- {selected_model} MLP modeli en iyi performansı gösterdi")
print(f"- En uygun aktivasyon fonksiyonu: {best_activation}")
print(f"- Final doğruluk değeri: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"- Scikit-learn MLP doğruluk: {accuracy_score(y_test, y):.4f}")
print(f"- PyTorch MLP doğruluk: {accuracy_score(y_test, y_pred_torch):.4f}")
print("\nProje tamamlandı! Sonuçlar ve grafikler kaydedildi.")