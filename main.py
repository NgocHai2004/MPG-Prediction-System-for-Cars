from Readfile import Read_file
from ydata_profiling import ProfileReport
def main():
    path = r"data/car.csv"
    data = Read_file(path).read()
    # report = ProfileReport(data,title ="Profiling Report")
    # report.to_file("report.html")
    print (data.columns)
    target = "mpg"
    y = data[target]
    x = data.drop([target,"car name"],axis=1)
    
    print(x)

main()
