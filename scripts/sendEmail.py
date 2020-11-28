import smtplib, ssl

class EmailAlert:
	smtp_server = "mailrelay.tugraz.at"
	port = 587  # For starttls
	sender_email = "martinwinter@tugraz.at"
	receiver_email = "Winter.Martin@live.at"
	password = "pA64LgBE7bkc!"

	def __init__(self):
		self.smtp_server = "mailrelay.tugraz.at"
		self.port = 587
		self.sender_email = "martinwinter@tugraz.at"
		self.receiver_email = "Winter.Martin@live.at"
		self.password = "pA64LgBE7bkc!"

	def sendAlert(self, message):
		context = ssl.create_default_context()
		with smtplib.SMTP(self.smtp_server, self.port) as server:
			server.ehlo()  # Can be omitted
			server.starttls(context=context)
			server.ehlo()  # Can be omitted
			server.login(self.sender_email, self.password)
			server.sendmail(self.sender_email, self.receiver_email, message)

if __name__ == "__main__":
	main()