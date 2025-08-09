# notify_failure.py
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg.set_content("Training script failed on host.")
msg["Subject"] = "Training Script Failed"
msg["From"] = "ellen660@csail.mit.edu"
msg["To"] = "ellen660@mit.edu"

s = smtplib.SMTP("outgoing.csail.mit.edu")  # Adjust if needed
s.send_message(msg)
s.quit()
print(f'Email sent to {msg["To"]}')