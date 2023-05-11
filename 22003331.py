
# importing requiredpackages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import
from sklearn import cluster


def read_file(path):
    """
    function takes name of the file and reads it from local 
    directory and loads this file into a dataframe. After that transposes the 
    dataframe and returns the original and transposed dataframes.

    Parameters
    ----------
    path : str
        a string containing path to the CSV datafile to be read.

    Returns
    -------
    datafile : pandas.DataFrame
        the resulting DataFrame read from the CSV file..

    """
    datafile = pd.read_csv(path)
    print(datafile)
    return datafile


file_path = r"C:\Users\User\Documents\assignment3\cl_change.csv"
datafile = read_file(file_path)
# Print the column names of the DataFrame
print(datafile.columns)

# Select indicators from the datafile
indi1 = datafile[datafile['Indicator Name'] == 'Population, total']
indi2 = datafile[datafile['Indicator Name'] ==
                 'CO2 emissions from solid fuel consumption (% of total)']
indi3 = datafile[datafile['Indicator Name'] ==
                 'Electric power consumption (kWh per capita)']
indi4 = datafile[datafile['Indicator Name']
                 == 'Agricultural land (% of land area)']

# Drop unnessory columns & replace NaN values with 0
indi1 = indi1.drop(["Indicator Code", "Indicator Name",
                   "Country Code", "2021", "2020"], axis=1)
indi1 = indi1.replace(np.NaN, 0)
# Select data for selected countries
s_country = ["Argentina", "Canada", "China", "United Kingdom", "Netherlands"]
data = indi1["Country Name"].isin(s_country)
indi1 = indi1[data]
print(indi1)

# Transpose the data & reset the index
d_trans = np.transpose(indi1)
d_trans = d_trans.reset_index()
# Renames the columns & drop the 1st row
d_trans = d_trans.rename(columns=d_trans.iloc[0])
d_trans = d_trans.drop(0, axis=0)
# Renames the country name to Year
d_trans = d_trans.rename(columns={"Country Name": "Year"})
print(d_trans)

# converts to numeric data
d_trans["Year"] = pd.to_numeric(d_trans["Year"])
d_trans["Argentina"] = pd.to_numeric(d_trans["Argentina"])
d_trans["Canada"] = pd.to_numeric(d_trans["Canada"])
d_trans["China"] = pd.to_numeric(d_trans["China"])
d_trans["United Kingdom"] = pd.to_numeric(d_trans["United Kingdom"])
d_trans["Netherlands"] = pd.to_numeric(d_trans["Netherlands"])
d_trans = d_trans.dropna()

# Drop unnessory columns & replace NaN values with 0
indi2 = indi2.drop(["Indicator Code", "Indicator Name",
                   "Country Code", "2021", "2020"], axis=1)
indi2 = indi2.replace(np.NaN, 0)
data2 = indi2["Country Name"].isin(s_country)
indi2 = indi2[data2]
print(indi2)
# Transpose the data & reset the index
d_trans2 = np.transpose(indi2)
d_trans2 = d_trans2.reset_index()
d_trans2 = d_trans2.rename(columns=d_trans2.iloc[0])
d_trans2 = d_trans2.drop(0, axis=0)
d_trans2 = d_trans2.rename(columns={"Country Name": "Year"})
print(d_trans2)
# converts to numeric data
d_trans2["Year"] = pd.to_numeric(d_trans2["Year"])
d_trans2["Argentina"] = pd.to_numeric(d_trans2["Argentina"])
d_trans2["Canada"] = pd.to_numeric(d_trans2["Canada"])
d_trans2["China"] = pd.to_numeric(d_trans2["China"])
d_trans2["United Kingdom"] = pd.to_numeric(d_trans2["United Kingdom"])
d_trans2["Netherlands"] = pd.to_numeric(d_trans2["Netherlands"])
d_trans2 = d_trans2.dropna()

# Drop unnessory columns & replace NaN values with 0
indi3 = indi3.drop(["Indicator Code", "Indicator Name",
                   "Country Code", "2021", "2020"], axis=1)
indi3 = indi3.replace(np.NaN, 0)
data3 = indi3["Country Name"].isin(s_country)
indi3 = indi3[data3]
print(indi3)
# Transpose the data & reset the index
d_trans3 = np.transpose(indi3)
d_trans3 = d_trans3.reset_index()
d_trans3 = d_trans3.rename(columns=d_trans3.iloc[0])
d_trans3 = d_trans3.drop(0, axis=0)
d_trans3 = d_trans3.rename(columns={"Country Name": "Year"})
# converts to numeric data
d_trans3["Year"] = pd.to_numeric(d_trans3["Year"])
d_trans3["Argentina"] = pd.to_numeric(d_trans3["Argentina"])
d_trans3["Canada"] = pd.to_numeric(d_trans3["Canada"])
d_trans3["China"] = pd.to_numeric(d_trans3["China"])
d_trans3["United Kingdom"] = pd.to_numeric(d_trans3["United Kingdom"])
d_trans3["Netherlands"] = pd.to_numeric(d_trans3["Netherlands"])
d_trans3 = d_trans3.dropna()
print(d_trans3)

# Drop unnessory columns & replace NaN values with 0
indi4 = indi4.drop(["Indicator Code", "Indicator Name",
                   "Country Code", "2021", "2020"], axis=1)
indi4 = indi4.replace(np.NaN, 0)
data4 = indi4["Country Name"].isin(s_country)
indi4 = indi4[data4]
print(indi4)
# Transpose the data & reset the index
d_trans4 = np.transpose(indi4)
d_trans4 = d_trans4.reset_index()
d_trans4 = d_trans4.rename(columns=d_trans4.iloc[0])
d_trans4 = d_trans4.drop(0, axis=0)
d_trans4 = d_trans4.rename(columns={"Country Name": "Year"})
# converts to numeric data
d_trans4["Year"] = pd.to_numeric(d_trans4["Year"])
d_trans4["Argentina"] = pd.to_numeric(d_trans4["Argentina"])
d_trans4["Canada"] = pd.to_numeric(d_trans4["Canada"])
d_trans4["China"] = pd.to_numeric(d_trans4["China"])
d_trans4["United Kingdom"] = pd.to_numeric(d_trans4["United Kingdom"])
d_trans4["Netherlands"] = pd.to_numeric(d_trans4["Netherlands"])
d_trans4 = d_trans4.dropna()
print(d_trans4)

# Create a dataframe for UK & assign data to respective columns
United_Kingdom = pd.DataFrame()
United_Kingdom["Year"] = d_trans["Year"]
United_Kingdom["Total_population"] = d_trans["United Kingdom"]
United_Kingdom["co2_emission"] = d_trans2["United Kingdom"]
United_Kingdom["Electric_power_consumption"] = d_trans3["United Kingdom"]
United_Kingdom["Agri_land"] = d_trans4["United Kingdom"]
United_Kingdom = United_Kingdom.iloc[1:55, :]

# Create a dataframe for Netherlands & assign data to respective columns
Netherlands = pd.DataFrame()
Netherlands["Year"] = d_trans["Year"]
Netherlands["Total_population"] = d_trans["Netherlands"]
Netherlands["co2_emission"] = d_trans2["Netherlands"]
Netherlands["Electric_power_consumption"] = d_trans3["Netherlands"]
Netherlands["Agri_land"] = d_trans4["Netherlands"]
Netherlands = Netherlands.iloc[1:55, :]


def set_mat(country):
    """
    This function takes in a pandas dataframe containing information about a 
    country and plots a scatter matrix of the variables in the dataframe. 
    It then displays the plot to the user.

    Parameters
    ----------
    country : pandas.DataFrame
        dataframe containing information about a country for the respective 
        index.

    Returns
    -------
    plot Scatter matrix for the indicator

    """
    pd.plotting.scatter_matrix(country, figsize=(14.0, 12.0))
    plt.tight_layout()
    plt.show()


# Calling setmat function for UK & Netherlands
set_mat(United_Kingdom)
set_mat(Netherlands)


# Reshape the columns of interest into 2D arrays
pt = np.array(United_Kingdom["Total_population"]).reshape(-1, 1)
ag = np.array(United_Kingdom["Agri_land"]).reshape(-1, 1)
co = np.array(United_Kingdom["co2_emission"]).reshape(-1, 1)
ep = np.array(United_Kingdom["Electric_power_consumption"]).reshape(-1, 1)

# concatenates the CO2 emissions and total population data
cl = np.concatenate((co, pt), axis=1)
# create KMeans object with 4 clusters
nc = 4
kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(cl)
# assignining the label
label = kmeans.labels_
# finding the centers for cluster
k_center = kmeans.cluster_centers_
col = ['co2_emission', 'Total_population']
labels = pd.DataFrame(label, columns=['Cluster ID'])
result = pd.DataFrame(cl, columns=col)
# concat result and labels
res = pd.concat((result, labels), axis=1)
# plotting the cluster
plt.figure(figsize=(7.0, 7.0))
plt.title("United Kingdom CO2 Emission vs Total Population")
plt.scatter(res["co2_emission"], res["Total_population"], c=label, cmap="cool")
plt.xlabel("CO2 Emission")
plt.ylabel("Total Population")
# plotting centers of clusters
plt.scatter(k_center[:, 0], k_center[:, 1], marker="*", c="black", s=150)
plt.show()

# Reshape the columns of interest into 2D arrays
p_t = np.array(Netherlands["Total_population"]).reshape(-1, 1)
a_g = np.array(Netherlands["Agri_land"]).reshape(-1, 1)
c_o = np.array(Netherlands["co2_emission"]).reshape(-1, 1)
e_p = np.array(Netherlands["Electric_power_consumption"]).reshape(-1, 1)

cl = np.concatenate((c_o, p_t), axis=1)
# create KMeans object with 4 clusters
nc = 4
kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(cl)
# assignining the label
label = kmeans.labels_
# finding the centers for cluster
k_center = kmeans.cluster_centers_
col = ['co2_emission', 'Total_population']
labels = pd.DataFrame(label, columns=['Cluster ID'])
result = pd.DataFrame(cl, columns=col)
# concat result and labels
res = pd.concat((result, labels), axis=1)
# plotting the cluster
plt.figure(figsize=(7.0, 7.0))
plt.title("Netherlands CO2 Emission vs Total Population")
plt.scatter(res["co2_emission"], res["Total_population"], c=label, cmap="cool")
plt.xlabel("CO2 Emission")
plt.ylabel("Total Population")
# plotting centers of clusters
plt.scatter(k_center[:, 0], k_center[:, 1], marker="*", c="black", s=150)
plt.show()

United_Kingdom["Normalized_CO2"] = United_Kingdom["co2_emission"] / \
    United_Kingdom["co2_emission"].abs().max()
United_Kingdom

Netherlands["Normalized_CO2"] = Netherlands["co2_emission"] / \
    Netherlands["co2_emission"].abs().max()
Netherlands


def err_ranges(x, func, param, sigma):
    """
    This function calculates the lower and upper limits for a given set of 
    parameters and their corresponding errors.

    Parameters
    ----------
    x : 1D numpy array
        input array to be passed to the function
    func : function object
        function to be fitted to the data.
    param : tuple
        array of parameters obtained from fitting.
    sigma : tuple
        array of errors corresponding to the parameters.

    Returns
    -------
    which gives the lower and upper bounds of the function at each x value

    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    # list to hold upper and lower limits for parameters
    uplow = []
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


def poly(u, s, h, a, m):
    """
    function takes in an input array u representing years from 1990 and returns
    polynomial function.

    Parameters
    ----------
    u : 
        input array.
    s,h,a,m : 
        representing the coefficients of a polynomial function.

    Returns
    -------
    returns the corresponding values of the polynomial function.

    """
    u = u - 1990
    pf = s + h*u + a*u**2 + m*u**3
    return pf


pfit, pocr = opt.curve_fit(
    poly, United_Kingdom["Year"], United_Kingdom["Normalized_CO2"])
print("Fit parameter", pfit)
# extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pocr))
# create extended year range
years = np.arange(1980, 2036)
# call function to calculate upper & lower limits with extrapolation
lower, upper = err_ranges(years, poly, pfit, sigmas)
United_Kingdom["poly"] = poly(United_Kingdom["Year"], *pfit)
plt.figure(figsize=(15, 8))
plt.title("Polynomial Fit for United_Kingdom")
plt.plot(United_Kingdom["Year"],
         United_Kingdom["Normalized_CO2"], label="data")
plt.plot(United_Kingdom["Year"], United_Kingdom["poly"], label="fit")
# plot error ranges with transparency
plt.fill_between(years, lower, upper, alpha=0.5,label="Prediction till 2036")
plt.legend(loc="upper right")
plt.show()

pf, pocr = opt.curve_fit(
    poly, Netherlands["Year"], Netherlands["Normalized_CO2"])
print("Fit parameter", pf)
# extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pocr))
# create extended year range
years = np.arange(1980, 2036)
# call function to calculate upper and lower limits
lower, upper = err_ranges(years, poly, pf, sigmas)
Netherlands["poly"] = poly(Netherlands["Year"], *pf)
plt.figure(figsize=(15, 8))
plt.title("Polynomial Fit for Netherlands")
plt.plot(Netherlands["Year"], Netherlands["Normalized_CO2"], label="data")
plt.plot(Netherlands["Year"], Netherlands["poly"], label="fit")
# plot error ranges with transparency
plt.fill_between(years, lower, upper, alpha=0.5, label="Prediction till 2036")
plt.legend(loc="upper right")
plt.show()


def cubic(u, a, b, c, d):
    """
    function takes in an input array u and returns cubic function 
    a*u^3 + b*u^2 + c*u + d
    """

    f = a*u**3 + b*u**2 + c*u + d
    return f


# plot the best fitting function & the conifidence range for United_Kingdom
param, covar = opt.curve_fit(
    cubic, United_Kingdom["Year"], United_Kingdom["Normalized_CO2"])
# create monotonic x-array for plotting
x = United_Kingdom["Year"]
y = cubic(United_Kingdom["Year"], *param)
y_pred = poly(United_Kingdom["Year"], *pfit)

# Calculate the confidence range
y_err = United_Kingdom["Normalized_CO2"] - y_pred
mse = np.mean(y_err ** 2)
n = len(x)
# 1.96 for 95% confidence interval
conf_interval = 1.96 * np.sqrt(mse / n)
plt.figure(figsize=(15, 8))

plt.plot(United_Kingdom["Year"], United_Kingdom["Normalized_CO2"],
         "o", markersize=3, label="Data")
plt.plot(x, y, color='red', label='Fitted Function')
# Labels for x and y axis
plt.xlabel("Years", fontweight='bold', fontsize=12)
plt.ylabel("co2_emission", fontweight='bold', fontsize=12)
plt.fill_between(United_Kingdom["Year"], y_pred - conf_interval, y_pred +
                 conf_interval, color='gray', alpha=0.5, label='Confidence Range')
plt.title(
    "Best fitting function and the conifidence range for United_Kingdom", fontsize=12)
plt.legend(fontsize=12)
plt.show()


# plot the best fitting function & the conifidence range for Netherlands
param, covar = opt.curve_fit(
    cubic, Netherlands["Year"], Netherlands["Normalized_CO2"])
# create monotonic x-array for plotting
x = Netherlands["Year"]
y = cubic(Netherlands["Year"], *param)
y_pred = poly(Netherlands["Year"], *pf)

# Calculate the confidence range
y_err = Netherlands["Normalized_CO2"] - y_pred
mse = np.mean(y_err ** 2)
n = len(x)
# 1.96 for 95% confidence interval
conf_interval = 1.96 * np.sqrt(mse / n)
# plotting the line graph
plt.figure(figsize=(15, 8))
plt.plot(Netherlands["Year"], Netherlands["Normalized_CO2"],
         "o", markersize=3, label="Data")
plt.plot(x, y, color='red', label='Fitted Function')
# Labels for x and y axis
plt.xlabel("Years", fontweight='bold', fontsize=12)
plt.ylabel("co2_emission", fontweight='bold', fontsize=12)
plt.fill_between(Netherlands["Year"], y_pred - conf_interval, y_pred +
                 conf_interval, color='gray', alpha=0.5, label='Confidence Range')
plt.title(
    "Best fitting function and the conifidence range for Netherlands", fontsize=12)
plt.legend(fontsize=12)
plt.show()
