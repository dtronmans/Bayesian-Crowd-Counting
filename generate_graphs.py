def train_array_from_txt(txt_path):
    loss, mse, mae = [], [], []
    with open(txt_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            temp_line = line.strip().split(" ")
            if len(temp_line) > 4 and temp_line[4] == "Train,":
                loss.append(float(temp_line[6].replace(",", "")))
                mse.append(float(temp_line[8].replace(",", "")))
                mae.append(float(temp_line[10].replace(",", "")))

    return loss, mse, mae


def val_array_from_txt(txt_path):
    mse, mae = [], []
    with open(txt_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            temp_line = line.strip().split(" ")
            print(temp_line)
            if len(temp_line) > 4 and temp_line[4] == "Val,":
                mse.append(float(temp_line[6].replace(",", "")))
                mae.append(float(temp_line[8].replace(",", "")))

    return mse, mae


if __name__ == "__main__":
    log_path = "train.log"
    loss, train_mse, train_mae = train_array_from_txt(log_path)
    val_mse, val_mae = val_array_from_txt(log_path)

    generate_graphs(loss, train_mse, train_mae, val_mse, val_mae)
