import plotly.graph_objects as go
import pdb

#gapminder = px.data.gapminder().query("country=='Canada'")
#fig = px.line(gapminder, x="year", y="lifeExp", title='Life expectancy in Canada')
#fig.show()


iterations = list(range(1, 13001))
trainLoss = [0] * 3
logfiles = ["log_Bosch_handmade_iter_0.log", "log_Bosch_pi_made_iter_0.log", "log_Bosch_pi_made_large_iter_0.log"]
lognames = ["Handmade", "Synthetic", "Synthetic large"]
total_logresults = []
#logfiles_banana = ["log_Banana_pi_made_iter_0.log", "log_Banana_coco_iter_0.log"]
for i, logfile in enumerate(logfiles):
    with open("/Users/andreas/Desktop/HCII/" + str(logfile), "r") as log:
        trainLoss = []
        for line in log:
            #pdb.set_trace()
            if line.startswith('S') or line.startswith('I') or line.startswith('T'):
                continue
            trainLoss.append(line.split(',')[2][:-1])
    print(len(trainLoss))
    total_logresults.append(trainLoss)
    #pdb.set_trace()
fig = go.Figure()
fig.add_trace(go.Scatter(x=iterations, y=total_logresults[0],
                    mode='lines',
                    name=lognames[0]))
fig.add_trace(go.Scatter(x=iterations, y=total_logresults[1],
                    mode='lines',
                    name=lognames[1]))
fig.add_trace(go.Scatter(x=iterations, y=total_logresults[2],
                    mode='lines',
                    name=lognames[2]))
fig.show()