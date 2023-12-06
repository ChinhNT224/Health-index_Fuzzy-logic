from flask import Flask, render_template, request
from flask import send_file
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

respiratory_rate = ctrl.Antecedent(np.arange(0, 40, 1), 'respiratory_rate')
heart_rate = ctrl.Antecedent(np.arange(0, 150, 1), 'heart_rate')
systolic_blood_pressure = ctrl.Antecedent(np.arange(0, 200, 1), 'systolic_blood_pressure')
vision = ctrl.Antecedent(np.arange(0, 10, 1), 'vision')
bmi = ctrl.Antecedent(np.arange(0, 40, 1), 'bmi')
health_index = ctrl.Consequent(np.arange(0, 100, 1), 'health_index')

def get_bmi_memberships(gender, age):
    if gender == 'male':
        if 18 <= age:
            return [0, 0, 10, 18], [15, 20, 25, 30], [28, 35, 40, 40]
    elif gender == 'female':
        if 18 <= age:
            return [0, 0, 10, 16], [14, 18.5, 25, 30], [28, 35, 40, 40]
    else:
        return [0, 0, 10, 18], [15, 20, 25, 30], [28, 35, 40, 40]

def get_respiratory_rate_memberships(gender, age):
    if gender == 'male':
        if 18 <= age < 50:
            return [0, 0, 5, 10], [8, 12, 20, 30], [25, 35, 40, 40]
        elif 50 <= age < 80:
            return [0, 0, 5, 10], [8, 12, 28, 30], [25, 35, 40, 40]
        elif 80 <= age:
            return [0, 0, 5, 10], [8, 10, 30, 30], [25, 35, 40, 40]
    elif gender == 'female':
        if 18 <= age < 50:
            return [0, 0, 5, 10], [8, 12, 20, 30], [25, 35, 40, 40]
        elif 50 <= age < 80:
            return [0, 0, 5, 10], [8, 12, 28, 30], [25, 35, 40, 40]
        elif 80 <= age:
            return [0, 0, 5, 10], [8, 10, 30, 30], [25, 35, 40, 40]
    else:
        return [0, 0, 5, 10], [8, 12, 20, 30], [25, 35, 40, 40]

def get_heart_rate_memberships(gender, age):
    if gender == 'male':
        if 2 <= age < 7:
            return [0, 0, 40, 70], [60, 75, 120, 130], [110, 140, 150, 150]
        elif 7 <= age < 18:
            return [0, 0, 40, 70], [60, 75, 110, 120], [100, 130, 150, 150]
        elif 18 <= age:
            return [0, 0, 40, 55], [50, 60, 100, 120], [90, 130, 150, 150]
    elif gender == 'female':
        if 2 <= age < 7:
            return [0, 0, 40, 70], [60, 75, 120, 130], [110, 140, 150, 150]
        elif 7 <= age < 18:
            return [0, 0, 40, 70], [60, 75, 110, 120], [100, 130, 150, 150]
        elif 18 <= age:
            return [0, 0, 40, 55], [50, 60, 100, 120], [90, 130, 150, 150]
    else:
        return [0, 0, 40, 70], [60, 75, 120, 130], [110, 140, 150, 150]

def get_systolic_blood_pressure_memberships(gender, age):
    if gender == 'male':
        if 1 <= age < 5:
            return [0, 0, 40, 70], [60, 80, 110, 130], [120, 140, 200, 200]
        elif 5 <= age < 13:
            return [0, 0, 40, 70], [60, 85, 120, 130], [120, 140, 200, 200]
        elif 13 <= age < 15:
            return [0, 0, 40, 70], [60, 95, 105, 130], [120, 140, 200, 200]
        elif 15 <= age < 19:
            return [0, 0, 40, 70], [60, 117, 120, 130], [120, 140, 200, 200]
        elif 19 <= age < 24:
            return [0, 0, 40, 70], [60, 108, 120, 132], [120, 140, 200, 200]
        elif 24 <= age < 29:
            return [0, 0, 40, 70], [60, 109, 121, 133], [125, 140, 200, 200]
        elif 29 <= age < 34:
            return [0, 0, 40, 70], [60, 110, 134, 140], [135, 144, 200, 200]
        elif 34 <= age < 39:
            return [0, 0, 40, 70], [60, 111, 135, 140], [136, 144, 200, 200]
        elif 39 <= age < 44:
            return [0, 0, 40, 70], [60, 125, 125, 140], [136, 144, 200, 200]
        elif 44 <= age < 50:
            return [0, 0, 40, 70], [60, 115, 139, 143], [140, 144, 200, 200]
        elif 50 <= age < 55:
            return [0, 0, 40, 70], [60, 116, 142, 145], [143, 150, 200, 200]
        elif 55 <= age < 60:
            return [0, 0, 40, 70], [60, 118, 144, 150], [146, 160, 200, 200]
        elif 60 <= age:
            return [0, 0, 40, 70], [60, 134, 134, 150], [140, 160, 200, 200]
    elif gender == 'female':
        if 1 <= age < 5:
            return [0, 0, 40, 70], [60, 80, 110, 130], [120, 140, 200, 200]
        elif 5 <= age < 13:
            return [0, 0, 40, 70], [60, 85, 120, 130], [120, 140, 200, 200]
        elif 13 <= age < 15:
            return [0, 0, 40, 70], [60, 95, 105, 130], [120, 140, 200, 200]
        elif 15 <= age < 19:
            return [0, 0, 40, 70], [60, 117, 120, 130], [120, 140, 200, 200]
        elif 19 <= age < 24:
            return [0, 0, 40, 70], [60, 108, 120, 132], [120, 140, 200, 200]
        elif 24 <= age < 29:
            return [0, 0, 40, 70], [60, 109, 121, 133], [125, 140, 200, 200]
        elif 29 <= age < 34:
            return [0, 0, 40, 70], [60, 110, 134, 140], [135, 144, 200, 200]
        elif 34 <= age < 39:
            return [0, 0, 40, 70], [60, 111, 135, 140], [136, 144, 200, 200]
        elif 39 <= age < 44:
            return [0, 0, 40, 70], [60, 125, 125, 140], [136, 144, 200, 200]
        elif 44 <= age < 50:
            return [0, 0, 40, 70], [60, 115, 139, 143], [140, 144, 200, 200]
        elif 50 <= age < 55:
            return [0, 0, 40, 70], [60, 116, 142, 145], [143, 150, 200, 200]
        elif 55 <= age < 60:
            return [0, 0, 40, 70], [60, 118, 144, 150], [146, 160, 200, 200]
        elif 60 <= age:
            return [0, 0, 40, 70], [60, 134, 134, 150], [140, 160, 200, 200]
    else:
        return [0, 0, 40, 70], [60, 108, 120, 132], [120, 140, 200, 200]

gender_input = 'male'
age_input = 25
bmi_membership_low, bmi_membership_medium, bmi_membership_high = get_bmi_memberships(gender_input, age_input)
respiratory_rate_membership_low, respiratory_rate_membership_medium, respiratory_rate_membership_high = get_respiratory_rate_memberships(gender_input, age_input)
heart_rate_membership_low, heart_rate_membership_medium, heart_rate_membership_high = get_heart_rate_memberships(gender_input, age_input)
systolic_blood_pressure_membership_low, systolic_blood_pressure_membership_medium, systolic_blood_pressure_membership_high = get_systolic_blood_pressure_memberships(gender_input, age_input)

bmi['low'] = fuzz.trapmf(bmi.universe, bmi_membership_low)
bmi['medium'] = fuzz.trapmf(bmi.universe, bmi_membership_medium)
bmi['high'] = fuzz.trapmf(bmi.universe, bmi_membership_high)

respiratory_rate['low'] = fuzz.trapmf(respiratory_rate.universe, respiratory_rate_membership_low)
respiratory_rate['medium'] = fuzz.trapmf(respiratory_rate.universe, respiratory_rate_membership_medium)
respiratory_rate['high'] = fuzz.trapmf(respiratory_rate.universe, respiratory_rate_membership_high)

heart_rate['low'] = fuzz.trapmf(heart_rate.universe, heart_rate_membership_low)
heart_rate['medium'] = fuzz.trapmf(heart_rate.universe, heart_rate_membership_medium)
heart_rate['high'] = fuzz.trapmf(heart_rate.universe, heart_rate_membership_high)

systolic_blood_pressure['low'] = fuzz.trapmf(systolic_blood_pressure.universe, systolic_blood_pressure_membership_low)
systolic_blood_pressure['medium'] = fuzz.trapmf(systolic_blood_pressure.universe, systolic_blood_pressure_membership_medium)
systolic_blood_pressure['high'] = fuzz.trapmf(systolic_blood_pressure.universe, systolic_blood_pressure_membership_high)

vision['low'] = fuzz.trapmf(vision.universe, [0, 0, 2, 4])
vision['medium'] = fuzz.trapmf(vision.universe, [3, 4, 6, 7])
vision['high'] = fuzz.trapmf(vision.universe, [5, 8, 10, 10])

health_index['danger'] = fuzz.trapmf(health_index.universe, [0, 0, 30, 40])
health_index['alarming'] = fuzz.trimf(health_index.universe, [30, 40, 50])
health_index['weak'] = fuzz.trimf(health_index.universe, [40, 50, 60])
health_index['normal'] = fuzz.trimf(health_index.universe, [50, 60, 70])
health_index['great'] = fuzz.trapmf(health_index.universe, [60, 70, 100, 100])

rule1 = ctrl.Rule(bmi['medium'] & respiratory_rate['medium'] & heart_rate['medium'] &
                  systolic_blood_pressure['medium'] & vision['high'], health_index['great'])

rule2 = ctrl.Rule((bmi['medium'] & respiratory_rate['medium'] & heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & ~vision['high']) |
                  (bmi['medium'] & respiratory_rate['medium'] & heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & vision['high']) |
                  (bmi['medium'] & respiratory_rate['medium'] & ~heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & vision['high']) |
                  (bmi['medium'] & ~respiratory_rate['medium'] & heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & vision['high']) |
                  (~bmi['medium'] & respiratory_rate['medium'] & heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & vision['high']), health_index['normal'])

rule3 = ctrl.Rule((bmi['medium'] & respiratory_rate['medium'] & heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & ~vision['high']) |
                  (bmi['medium'] & respiratory_rate['medium'] & ~heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & ~vision['high']) |
                  (bmi['medium'] & ~respiratory_rate['medium'] & heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & ~vision['high']) |
                  (~bmi['medium'] & respiratory_rate['medium'] & heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & ~vision['high']) |
                  (bmi['medium'] & respiratory_rate['medium'] & ~heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & vision['high']) |
                  (bmi['medium'] & respiratory_rate['medium'] & heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & ~vision['high']) |
                  (bmi['medium'] & respiratory_rate['medium'] & ~heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & ~vision['high']) |
                  (bmi['medium'] & ~respiratory_rate['medium'] & heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & ~vision['high']) |
                  (~bmi['medium'] & respiratory_rate['medium'] & heart_rate['medium'] &
                  systolic_blood_pressure['medium'] & ~vision['high']), health_index['weak'])

rule4 = ctrl.Rule((bmi['medium'] & respiratory_rate['medium'] & ~heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & ~vision['high']) |
                  (bmi['medium'] & ~respiratory_rate['medium'] & heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & ~vision['high']) |
                  (~bmi['medium'] & respiratory_rate['medium'] & heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & ~vision['high']) |
                  (bmi['medium'] & ~respiratory_rate['medium'] & ~heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & vision['high']) |
                  (~bmi['medium'] & respiratory_rate['medium'] & ~heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & vision['high']) |
                  (~bmi['medium'] & ~respiratory_rate['medium'] & heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & vision['high']) |
                  (~bmi['medium'] & ~respiratory_rate['medium'] & ~heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & vision['high']), health_index['alarming'])

rule5 = ctrl.Rule((bmi['medium'] & ~respiratory_rate['medium'] & ~heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & ~vision['high']) |
                  (~bmi['medium'] & respiratory_rate['medium'] & ~heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & ~vision['high']) |
                  (~bmi['medium'] & ~respiratory_rate['medium'] & ~heart_rate['medium'] &
                   systolic_blood_pressure['medium'] & ~vision['high']) |
                  (~bmi['medium'] & ~respiratory_rate['medium'] & ~heart_rate['medium'] &
                   ~systolic_blood_pressure['medium'] & vision['high']) |
                  (~bmi['medium'] & ~respiratory_rate['medium'] & ~heart_rate['medium'] &
                  ~systolic_blood_pressure['medium'] & ~vision['high']), health_index['danger'])

health_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])

health_simulation = ctrl.ControlSystemSimulation(health_ctrl)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender_input = request.form['gender']
        age_input = int(request.form['age'])
        bmi_input = float(request.form['bmi'])
        respiratory_rate_input = float(request.form['respiratory_rate'])
        heart_rate_input = float(request.form['heart_rate'])
        systolic_blood_pressure_input = float(request.form['systolic_blood_pressure'])
        vision_input = float(request.form['vision'])

        health_simulation.input['bmi'] = bmi_input
        health_simulation.input['respiratory_rate'] = respiratory_rate_input
        health_simulation.input['heart_rate'] = heart_rate_input
        health_simulation.input['systolic_blood_pressure'] = systolic_blood_pressure_input
        health_simulation.input['vision'] = vision_input

        health_simulation.compute()

        predicted_health_index = health_simulation.output['health_index']

        if predicted_health_index < 40:
            predicted_health_category = "Nguy hiểm"
        elif 40 <= predicted_health_index < 50:
            predicted_health_category = "Báo động"
        elif 50 <= predicted_health_index < 60:
            predicted_health_category = "Sức khỏe yếu"
        elif 60 <= predicted_health_index < 70:
            predicted_health_category = "Sức khỏe bình thường"
        else:
            predicted_health_category = "Sức khỏe tốt"

        result_messages = []
        result_messages.append(predicted_health_index)
        result_messages.append(predicted_health_category)

        return render_template('result.html', result=result_messages)

    except ValueError:
        error_message = "Please enter valid numerical values for all health parameters."
        return render_template('error.html', error=error_message)

@app.route('/health_index_image')
def health_index_image():
    image_path = 'images/health_index_image.png'

    if not os.path.exists('images'):
        os.makedirs('images')

    plt.figure()
    bmi_image_path = 'images/bmi_plot.png'
    plt.figure()
    bmi.view()
    plt.savefig(bmi_image_path)
    plt.close()

    respiratory_rate_image_path ='images/respiratory_rate_plot.png'
    plt.figure()
    respiratory_rate.view()
    plt.savefig(respiratory_rate_image_path)
    plt.close()

    heart_rate_image_path = 'images/heart_rate_plot.png'
    plt.figure()
    heart_rate.view()
    plt.savefig(heart_rate_image_path)
    plt.close()

    systolic_blood_pressure_image_path = 'images/systolic_blood_pressure_plot.png'
    plt.figure()
    systolic_blood_pressure.view()
    plt.savefig(systolic_blood_pressure_image_path)
    plt.close()

    # Vision Plot
    vision_image_path = 'images/vision_plot.png'
    plt.figure()
    vision.view()
    plt.savefig(vision_image_path)
    plt.close()

    health_index.view(sim=health_simulation)
    plt.show()
    plt.savefig(image_path)
    plt.close()
    return send_file(image_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
