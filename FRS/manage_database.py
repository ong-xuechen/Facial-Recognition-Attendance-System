import sqlite3

# Connect to the database
conn = sqlite3.connect('broadcast_messages.db')
cursor = conn.cursor()

# Specify the ID value you want to delete
id_to_delete = 2  # Replace this with the desired ID value

# Delete the row with the specified ID value
cursor.execute("DELETE FROM broadcast_messages WHERE id=?", (id_to_delete,))
conn.commit()
print("Row with ID {} deleted successfully.".format(id_to_delete))

# Close the connection
conn.close()
