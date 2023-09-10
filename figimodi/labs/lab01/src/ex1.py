f = open('score.txt', 'r')

athlets = []
countries= {}

for line in f:
    grades = []
    sum = 0
    fields = line.split(' ')
    nome = fields[0]
    cognome = fields[1]
    country = fields[2]
    
    for i in range(3, 8):
        grades.append(float(fields[i]))
    
    grades.sort()
    grades = grades[1:-1]

    for grade in grades:
        sum += grade

    athlets.append({"nome":nome, "cognome":cognome, "country":country, "grade":sum})
    if (country in countries.keys()):
        countries[country] = float(countries[country]) + sum
    else:
        countries.update({country:sum})

athlets.sort(key=lambda x: x["grade"], reverse=True)
athlets = athlets[0:3]    

max = max(countries.values())

print('final ranking:')
for i in range(3):
    print(i+1, ": ", athlets[i]["nome"], " ", athlets[i]["cognome"], "Score: ", athlets[i]["grade"])

print("Best Country:")
for c in countries:
    if countries[c] == max:
        print(c, " Total Score: ", countries[c])