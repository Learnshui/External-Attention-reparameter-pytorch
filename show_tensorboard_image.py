#具体看transUnet
if iter_num % 20 == 0:
    image = image_batch[1, 0:1, :, :]
    image = (image - image.min()) / (image.max() - image.min())
    writer.add_image('train/Image', image, iter_num)
    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
    labs = label_batch[1, ...].unsqueeze(0) * 50
    writer.add_image('train/GroundTruth', labs, iter_num)
