from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import pairwise


app = Flask(__name__)

# Tải mô hình đã huấn luyện
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận thông tin từ form
    attack = float(request.form['attack'])
    base_egg_steps = float(request.form['base_egg_steps'])
    base_total = float(request.form['base_total'])
    defense = float(request.form['defense'])
    experience_growth = float(request.form['experience_growth'])
    height_m = float(request.form['height_m'])
    hp = float(request.form['hp'])
    weight_kg = float(request.form['weight_kg'])
    sp_attack = float(request.form['sp_attack'])
    sp_defense = float(request.form['sp_defense'])
    speed = float(request.form['speed'])
    tot_abilities = float(request.form['tot_abilities'])

    df_pokemon = pd.read_csv('df_resampled1.csv')
    
    epsilon = 0.2  # Sai số cho phép

    matching_pokemon = df_pokemon[
        (df_pokemon['attack'].between(attack - epsilon, attack + epsilon)) &
        (df_pokemon['base_egg_steps'].between(base_egg_steps - epsilon, base_egg_steps + epsilon)) &
        (df_pokemon['base_total'].between(base_total - epsilon, base_total + epsilon)) &
        (df_pokemon['defense'].between(defense - epsilon, defense + epsilon)) &
        (df_pokemon['experience_growth'].between(experience_growth - epsilon, experience_growth + epsilon)) &
        (df_pokemon['height_m'].between(height_m - epsilon, height_m + epsilon)) &
        (df_pokemon['hp'].between(hp - epsilon, hp + epsilon)) &
        (df_pokemon['weight_kg'].between(weight_kg - epsilon, weight_kg + epsilon)) &
        (df_pokemon['sp_attack'].between(sp_attack - epsilon, sp_attack + epsilon)) &
        (df_pokemon['sp_defense'].between(sp_defense - epsilon, sp_defense + epsilon)) &
        (df_pokemon['speed'].between(speed - epsilon, speed + epsilon)) &
        (df_pokemon['tot_abilities'].between(tot_abilities - epsilon, tot_abilities + epsilon))
]


    if not matching_pokemon.empty:
        # Nếu tìm thấy Pokémon khớp
        predicted_name = matching_pokemon['name'].values[0]
        predicted_type = matching_pokemon['type'].values[0]
        predicted_generation = matching_pokemon['generation'].values[0]
        
        is_legendary = "Yes" if matching_pokemon['is_legendary'].values[0] == 1 else "No"
    else:
        # Nếu không tìm thấy, tìm Pokémon tương tự
        input_data = np.array([[attack, base_egg_steps, base_total, defense, experience_growth,
                                height_m, hp, weight_kg, sp_attack, sp_defense, speed, tot_abilities
                                ]])

        # Chuẩn hóa dữ liệu đầu vào
        input_scaled = scaler.transform(input_data)

        # Tính khoảng cách
        similar_pokemon = df_pokemon[['attack', 'base_egg_steps', 'base_total', 'defense', 
                                       'experience_growth', 'height_m', 'hp', 'weight_kg', 
                                       'sp_attack', 'sp_defense', 'speed', 'tot_abilities'
                                      ]]

        distances = pairwise.euclidean_distances(input_scaled, similar_pokemon)

        # Tìm Pokémon gần nhất
        closest_indices = np.argsort(distances.flatten())[:5]  # Lấy 5 Pokémon gần nhất
        closest_pokemons = df_pokemon.iloc[closest_indices]
        
        predicted_name = ", ".join(closest_pokemons['name'].values)
        predicted_type = ", ".join(closest_pokemons['type'].values)
        predicted_generation = ", ".join(closest_pokemons['generation'].astype(str).values)
        is_legendary = ", ".join(["Yes" if x == 1 else "No" for x in closest_pokemons['is_legendary']])

    return render_template('result.html', is_legendary=is_legendary, predicted_name=predicted_name, predicted_type=predicted_type, predicted_generation=predicted_generation )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
