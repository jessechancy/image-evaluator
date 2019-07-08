# Steps to run on AWS EC2 instances
Notes: Amazon EC2 instances operate on Amazon linux, and so to install any packages, instead of using `apt-get`, we use `yum`
* After SSHing to the EC2 instance, we first need to update/install all the dependencies required
  * Update `python` version to v3
  `sudo yum install python3`
  * Install pip
  `#Download get-pip to current directory. It won't install anything, as of now
    curl -O https://bootstrap.pypa.io/get-pip.py

    #Use python3 to install pip
    python3 get-pip.py
    #this will install pip3 and pip3.6`
* We'd also need to change the google-chrome setting on the crawler to headless because we're running all of this on CLI
  * To do this, go `inscrawler_threading.py` file > line 39 > change the last argument to `False`
  * This line: `process = Thread(target=crawl_wrapper, args=[q, None, True, True])`
  * To this line: `process = Thread(target=crawl_wrapper, args=[q, None, True, False])`
  
Links
[https://devopsqa.wordpress.com/2018/03/08/install-google-chrome-and-chromedriver-in-amazon-linux-machine/]

Angelica
* Write up this ^, including EC2 instances
* continue trying AWS
  * Ensure that you can get at least all pics of a user
  * Try saving into one common storage space
  * Try deploying on multiple instances

Jesse
* Rotate IP address
* Make AWS account
