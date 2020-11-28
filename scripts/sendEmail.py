import smtplib, ssl

class EmailAlert:
	smtp_server = "mailrelay.tugraz.at"
	port = 587  # For starttls
	sender_email = "martinwinter@tugraz.at"
	receiver_email = "Winter.Martin@live.at"
	password = "test"

	def __init__(self, password, smtp_server="mailrelay.tugraz.at", port=587, sender_email="martinwinter@tugraz.at", receiver_email="Winter.Martin@live.at"):
		self.smtp_server = smtp_server
		self.port = port
		self.sender_email = sender_email
		self.receiver_email = receiver_email
		self.password = password

	def sendAlert(self, message):
		context = ssl.create_default_context()
		with smtplib.SMTP(self.smtp_server, self.port) as server:
			server.ehlo()  # Can be omitted
			server.starttls(context=context)
			server.ehlo()  # Can be omitted
			server.login(self.sender_email, self.password)
			server.sendmail(self.sender_email, self.receiver_email, message)