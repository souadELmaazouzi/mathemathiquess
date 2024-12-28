import xml.etree.ElementTree as ET
from ortools.sat.python import cp_model
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parser les données XML
def parse_xml(file_path):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return [], []

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        rooms = []
        for room in root.findall('rooms/room'):
            rooms.append({
                'id': room.get('id'),
                'capacity': int(room.get('capacity')),
            })

        classes = []
        for cls in root.findall('classes/class'):
            preferences = []
            for time in cls.findall('time'):
                preferences.append({
                    'start': int(time.get('start')),
                })
            class_limit = cls.get('classLimit', 0)
            classes.append({
                'id': cls.get('id'),
                'limit': int(class_limit),
                'preferences': preferences,
            })

        print(f"Salles chargées : {len(rooms)}")
        print(f"Cours chargés : {len(classes)}")
        return rooms, classes

    except ET.ParseError as e:
        print(f"Erreur lors du parsing du fichier XML : {e}")
        return [], []

# Construire le modèle
def build_model(rooms, classes):
    model = cp_model.CpModel()
    assignments = {}
    
    # Ajouter les variables d'affectation
    for cls in classes:
        for room in rooms:
            for time in cls['preferences']:
                if cls['limit'] <= room['capacity']:
                    key = (cls['id'], room['id'], time['start'])
                    assignments[key] = model.NewBoolVar(f"x_{key}")

        # Ajout pour salle fictive
        for time in cls['preferences']:
            key = (cls['id'], 'default', time['start'])
            assignments[key] = model.NewBoolVar(f"x_{cls['id']}_default_{time['start']}")

    # Contraintes : chaque cours doit être assigné
    for cls in classes:
        valid_assignments = [assignments[key] for key in assignments if key[0] == cls['id']]
        if valid_assignments:
            model.Add(sum(valid_assignments) == 1)

    # Contraintes : capacité des salles
    for room in rooms:
        for time_start in set(tp['start'] for cls in classes for tp in cls['preferences']):
            valid_assignments = [assignments[(cls['id'], room['id'], time_start)]
                                 for cls in classes
                                 if (cls['id'], room['id'], time_start) in assignments]
            if valid_assignments:
                model.Add(sum(valid_assignments) <= room['capacity'])

    # Minimiser l'utilisation de la salle fictive
    penalty = [assignments[key] for key in assignments if key[1] == 'default']
    model.Minimize(sum(penalty))

    return model, assignments

# Résolution du modèle
def solve_model(model, assignments):
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
        print("Solution trouvée.")
        results = {}
        for key, var in assignments.items():
            if solver.Value(var):
                cls, room, time = key
                if (room, time) not in results:
                    results[(room, time)] = []
                results[(room, time)].append(cls)
        return results
    else:
        print("Aucune solution trouvée.")
        return None

# Visualiser les résultats
def visualize_results(results):
    if not results:
        print("Aucun résultat à visualiser.")
        return

    data = []
    for (room, time), classes in results.items():
        for cls in classes:
            data.append({'Class': cls, 'Room': room, 'Time': time})

    df = pd.DataFrame(data)
    print(df)

    plt.figure(figsize=(10, 6))
    for room, group in df.groupby('Room'):
        plt.scatter(group['Time'], [room] * len(group), label=f"Salle {room}", s=100)
    plt.xlabel("Créneaux horaires")
    plt.ylabel("Salles")
    plt.title("Affectation des cours aux salles et créneaux")
    plt.legend()
    plt.show()

# Script principal
def main():
    file_path = "/Users/souad/Desktop/mathemathiquess/data/pu-spr07-llr.xml"

    rooms, classes = parse_xml(file_path)
    if not rooms or not classes:
        return

    # Réduction pour tester avec un sous-ensemble
    classes = classes  # Retirer : classes = classes[:10]
    rooms = rooms  # Retirer : rooms = rooms[:5]
# Limitez à 5 salles

    model, assignments = build_model(rooms, classes)
    results = solve_model(model, assignments)

    if results:
        visualize_results(results)
    else:
        print("Impossible de trouver une solution.")

if __name__ == "__main__":
    main()
