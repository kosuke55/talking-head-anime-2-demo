from importlib.resources import is_resource
import logging
from multiprocessing.dummy import current_process
import os
import sys
from typing import List

sys.path.append(os.getcwd())

import numpy
import torch
import wx
import PIL.Image
import cv2
import dlib

from tha2.poser.poser import Poser, PoseParameterCategory, PoseParameterGroup
from tha2.util import extract_PIL_image_from_filelike, resize_PIL_image, extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy, cv2pil

from puppet.head_pose_solver import HeadPoseSolver
from puppet.util import compute_left_eye_normalized_ratio, compute_right_eye_normalized_ratio, \
    compute_mouth_normalized_ratio

class MorphCategoryControlPanel(wx.Panel):
    def __init__(self,
                 parent,
                 title: str,
                 pose_param_category: PoseParameterCategory,
                 param_groups: List[PoseParameterGroup]):
        super().__init__(parent, style=wx.SIMPLE_BORDER)
        self.pose_param_category = pose_param_category
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)

        title_text = wx.StaticText(self, label=title, style=wx.ALIGN_CENTER)
        self.sizer.Add(title_text, 0, wx.EXPAND)

        self.param_groups = [group for group in param_groups if group.get_category() == pose_param_category]
        self.choice = wx.Choice(self, choices=[group.get_group_name() for group in self.param_groups])
        if len(self.param_groups) > 0:
            self.choice.SetSelection(0)
        self.choice.Bind(wx.EVT_CHOICE, self.on_choice_updated)
        self.sizer.Add(self.choice, 0, wx.EXPAND)

        self.left_slider = wx.Slider(self, minValue=-1000, maxValue=1000, value=-1000, style=wx.HORIZONTAL)
        self.sizer.Add(self.left_slider, 0, wx.EXPAND)

        self.right_slider = wx.Slider(self, minValue=-1000, maxValue=1000, value=-1000, style=wx.HORIZONTAL)
        self.sizer.Add(self.right_slider, 0, wx.EXPAND)

        self.checkbox = wx.CheckBox(self, label="Show")
        self.checkbox.SetValue(True)
        self.sizer.Add(self.checkbox, 0, wx.SHAPED | wx.ALIGN_CENTER)

        self.update_ui()

        self.sizer.Fit(self)

    def update_ui(self):
        param_group = self.param_groups[self.choice.GetSelection()]
        if param_group.is_discrete():
            self.left_slider.Enable(False)
            self.right_slider.Enable(False)
            self.checkbox.Enable(True)
        elif param_group.get_arity() == 1:
            self.left_slider.Enable(True)
            self.right_slider.Enable(False)
            self.checkbox.Enable(False)
        else:
            self.left_slider.Enable(True)
            self.right_slider.Enable(True)
            self.checkbox.Enable(False)

    def on_choice_updated(self, event: wx.Event):
        param_group = self.param_groups[self.choice.GetSelection()]
        if param_group.is_discrete():
            self.checkbox.SetValue(True)
        self.update_ui()

    def set_param_value(self, pose: List[float]):
        if len(self.param_groups) == 0:
            return
        selected_morph_index = self.choice.GetSelection()
        param_group = self.param_groups[selected_morph_index]
        param_index = param_group.get_parameter_index()
        if param_group.is_discrete():
            if self.checkbox.GetValue():
                for i in range(param_group.get_arity()):
                    pose[param_index + i] = 1.0
        else:
            param_range = param_group.get_range()
            alpha = (self.left_slider.GetValue() + 1000) / 2000.0
            pose[param_index] = param_range[0] + (param_range[1] - param_range[0]) * alpha
            if param_group.get_arity() == 2:
                alpha = (self.right_slider.GetValue() + 1000) / 2000.0
                pose[param_index + 1] = param_range[0] + (param_range[1] - param_range[0]) * alpha


class RotationControlPanel(wx.Panel):
    def __init__(self, parent,
                 pose_param_category: PoseParameterCategory,
                 param_groups: List[PoseParameterGroup]):
        super().__init__(parent, style=wx.SIMPLE_BORDER)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)

        self.param_groups = [group for group in param_groups if group.get_category() == pose_param_category]
        for param_group in self.param_groups:
            assert not param_group.is_discrete()
            assert param_group.get_arity() == 1

        self.sliders = []
        for param_group in self.param_groups:
            static_text = wx.StaticText(self, label="--- %s ---" % param_group.get_group_name(), style=wx.ALIGN_CENTER)
            self.sizer.Add(static_text, 0, wx.EXPAND)
            slider = wx.Slider(self, minValue=-1000, maxValue=1000, value=0, style=wx.HORIZONTAL)
            self.sizer.Add(slider, 0, wx.EXPAND)
            self.sliders.append(slider)

        self.sizer.Fit(self)

    def set_param_value(self, pose: List[float]):
        if len(self.param_groups) == 0:
            return
        for param_group_index in range(len(self.param_groups)):
            param_group = self.param_groups[param_group_index]
            slider = self.sliders[param_group_index]
            param_range = param_group.get_range()
            param_index = param_group.get_parameter_index()
            alpha = (slider.GetValue() + 1000) / 2000.0
            pose[param_index] = param_range[0] + (param_range[1] - param_range[0]) * alpha


class MainFrame(wx.Frame):
    def __init__(self, poser: Poser,
                 face_detector,
                 landmark_locator,
                 video_capture,
                 device: torch.device):
        super().__init__(None, wx.ID_ANY, "Poser")
        self.poser = poser
        self.face_detector = face_detector
        self.landmark_locator = landmark_locator
        self.video_capture = video_capture
        self.device = device

        self.wx_source_image = None
        self.wx_source_video_frame = None
        self.head_pose_solver = HeadPoseSolver()

        self.torch_source_image = None
        self.source_image_string = "Nothing yet!"

        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)
        self.init_left_panel()
        self.init_control_panel()
        self.init_right_panel()
        self.main_sizer.Fit(self)

        self.timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_result_image_panel, self.timer)

        save_image_id = wx.NewIdRef()
        self.Bind(wx.EVT_MENU, self.on_save_image, id=save_image_id)
        accelerator_table = wx.AcceleratorTable([
            (wx.ACCEL_CTRL, ord('S'), save_image_id)
        ])
        self.SetAcceleratorTable(accelerator_table)

        # self.pose_size = 6  # face_detector can detect only 6 poses

        self.last_pose = None
        self.detected_last_pose = None
        self.detected_pose = None
        self.last_output_index = self.output_index_choice.GetSelection()
        self.last_output_numpy_image = None

    def init_left_panel(self):
        self.left_panel = wx.Panel(self, style=wx.SIMPLE_BORDER)
        left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.left_panel.SetSizer(left_panel_sizer)
        self.left_panel.SetAutoLayout(1)

        self.source_image_panel = wx.Panel(self.left_panel, size=(256, 256), style=wx.SIMPLE_BORDER)
        self.source_image_panel.Bind(wx.EVT_PAINT, self.paint_source_image_panel)
        left_panel_sizer.Add(self.source_image_panel, 0, wx.FIXED_MINSIZE)

        self.load_image_button = wx.Button(self.left_panel, wx.ID_ANY, "\nLoad Image\n\n")
        left_panel_sizer.Add(self.load_image_button, 1, wx.EXPAND)
        self.load_image_button.Bind(wx.EVT_BUTTON, self.load_image)

        # self.source_video_panel = wx.Panel(self.left_panel, size=(256, 256), style=wx.SIMPLE_BORDER)
        # left_panel_sizer.Add(self.source_video_panel, 0, wx.FIXED_MINSIZE)

        left_panel_sizer.Fit(self.left_panel)
        self.main_sizer.Add(self.left_panel, 0, wx.FIXED_MINSIZE)

    def init_control_panel(self):
        self.control_panel = wx.Panel(self, style=wx.SIMPLE_BORDER, size=(256, 256))
        self.control_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.source_video_panel = wx.Panel(self.control_panel, size=(256, 256), style=wx.SIMPLE_BORDER)
        self.control_panel_sizer.Add(self.source_video_panel, 0, wx.FIXED_MINSIZE)
        self.control_panel.SetAutoLayout(1)
        self.control_panel_sizer.Fit(self.left_panel)
        self.main_sizer.Add(self.control_panel, 0, wx.FIXED_MINSIZE)

        # self.control_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        # self.control_panel.SetSizer(self.control_panel_sizer)


        # morph_categories = [
        #     PoseParameterCategory.EYEBROW,
        #     PoseParameterCategory.EYE,
        #     PoseParameterCategory.MOUTH,
        #     PoseParameterCategory.IRIS_MORPH
        # ]
        # morph_category_titles = {
        #     PoseParameterCategory.EYEBROW: "--- Eyebrow ---",
        #     PoseParameterCategory.EYE: "--- Eye ---",
        #     PoseParameterCategory.MOUTH: "--- Mouth ---",
        #     PoseParameterCategory.IRIS_MORPH: "--- Iris morphs ---",
        # }
        # self.morph_control_panels = {}
        # for category in morph_categories:
        #     param_groups = self.poser.get_pose_parameter_groups()
        #     filtered_param_groups = [group for group in param_groups if group.get_category() == category]
        #     if len(filtered_param_groups) == 0:
        #         continue
        #     control_panel = MorphCategoryControlPanel(
        #         self.control_panel,
        #         morph_category_titles[category],
        #         category,
        #         self.poser.get_pose_parameter_groups())
        #     self.morph_control_panels[category] = control_panel
        #     self.control_panel_sizer.Add(control_panel, 0, wx.EXPAND)

        # self.rotation_control_panels = {}
        # rotation_categories = [
        #     PoseParameterCategory.IRIS_ROTATION,
        #     PoseParameterCategory.FACE_ROTATION
        # ]
        # for category in rotation_categories:
        #     param_groups = self.poser.get_pose_parameter_groups()
        #     filtered_param_groups = [group for group in param_groups if group.get_category() == category]
        #     if len(filtered_param_groups) == 0:
        #         continue
        #     control_panel = RotationControlPanel(
        #         self.control_panel,
        #         category,
        #         self.poser.get_pose_parameter_groups())
        #     self.rotation_control_panels[category] = control_panel
        #     self.control_panel_sizer.Add(control_panel, 0, wx.EXPAND)

        # self.control_panel_sizer.Fit(self.control_panel)
        # self.main_sizer.Add(self.control_panel, 1, wx.EXPAND)

    def init_right_panel(self):
        self.right_panel = wx.Panel(self, style=wx.SIMPLE_BORDER)
        right_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_panel.SetSizer(right_panel_sizer)
        self.right_panel.SetAutoLayout(1)

        self.result_image_panel = wx.Panel(self.right_panel, size=(256, 256), style=wx.SIMPLE_BORDER)
        self.result_image_panel.Bind(wx.EVT_PAINT, self.paint_result_image_panel)
        self.output_index_choice = wx.Choice(
            self.right_panel,
            choices=[str(i) for i in range(self.poser.get_output_length())])
        self.output_index_choice.SetSelection(0)
        right_panel_sizer.Add(self.result_image_panel, 0, wx.FIXED_MINSIZE)
        right_panel_sizer.Add(self.output_index_choice, 0, wx.EXPAND)

        self.save_image_button = wx.Button(self.right_panel, wx.ID_ANY, "\nSave Image\n\n")
        right_panel_sizer.Add(self.save_image_button, 1, wx.EXPAND)
        self.save_image_button.Bind(wx.EVT_BUTTON, self.on_save_image)

        right_panel_sizer.Fit(self.right_panel)
        self.main_sizer.Add(self.right_panel, 0, wx.FIXED_MINSIZE)

    def create_param_category_choice(self, param_category: PoseParameterCategory):
        params = []
        for param_group in self.poser.get_pose_parameter_groups():
            if param_group.get_category() == param_category:
                params.append(param_group.get_group_name())
        choice = wx.Choice(self.control_panel, choices=params)
        if len(params) > 0:
            choice.SetSelection(0)
        return choice

    def load_image(self, event: wx.Event):
        dir_name = "data/illust"
        file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            pil_image = resize_PIL_image(extract_PIL_image_from_filelike(image_file_name))
            w, h = pil_image.size
            if pil_image.mode != 'RGBA':
                self.source_image_string = "Image must have alpha channel!"
                self.wx_source_image = None
                self.torch_source_image = None
            else:
                self.wx_source_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image).to(self.device)
                # dc = wx.PaintDC(self.source_image_panel)
                # dc.Clear()
                # dc.DrawBitmap(self.wx_source_image, 0, 0, True)

            self.Refresh()
        file_dialog.Destroy()

    def paint_source_image_panel(self, event: wx.Event):
        if self.wx_source_image is None:
            self.draw_source_image_string(self.source_image_panel, use_paint_dc=True)
        else:
            dc = wx.PaintDC(self.source_image_panel)
            dc.Clear()
            dc.DrawBitmap(self.wx_source_image, 0, 0, True)

    def paint_result_image_panel(self, event: wx.Event):
        self.last_pose = None

    def draw_source_image_string(self, widget, use_paint_dc: bool = True):
        if use_paint_dc:
            dc = wx.PaintDC(widget)
        else:
            dc = wx.ClientDC(widget)
        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent(self.source_image_string)
        dc.DrawText(self.source_image_string, 128 - w // 2, 128 - h // 2)

    def get_current_pose(self):
        current_pose = [0.0 for i in range(self.poser.get_num_parameters())]
        # for morph_control_panel in self.morph_control_panels.values():
        #     # print("morph_control_panel ", morph_control_panel)
        #     morph_control_panel.set_param_value(current_pose)
        #     # print(current_pose)
        # for rotation_control_panel in self.rotation_control_panels.values():
        #     # print("rotation_control_panel", rotation_control_panel)
        #     rotation_control_panel.set_param_value(current_pose)

        if self.detected_pose is not None:
            current_pose[12] = self.detected_pose[4]  # right eye
            current_pose[13] = self.detected_pose[3]  # left eye
            current_pose[26] = self.detected_pose[5]
            current_pose[39] = self.detected_pose[0]
            current_pose[40] = self.detected_pose[1]
            current_pose[41] = self.detected_pose[2]

        for i in range(len(current_pose)):
            print(i, current_pose[i])

        return current_pose

    def update_result_image_panel(self, event: wx.Event):
        _, frame = self.video_capture.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(rgb_frame)
        euler_angles = None
        face_landmarks = None
        if len(faces) > 0:
            face_rect = faces[0]
            face_landmarks = self.landmark_locator(rgb_frame, face_rect)
            face_box_points, euler_angles = self.head_pose_solver.solve_head_pose(
                face_landmarks)
            self.draw_face_landmarks(rgb_frame, face_landmarks)
            self.draw_face_box(rgb_frame, face_box_points)

        # visualize land mark and pose
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        pil_image = resize_PIL_image(cv2pil(bgr_frame))
        pil_image.putalpha(255)
        w, h = pil_image.size
        if pil_image.mode != 'RGBA':
            frame = None
        else:
            frame = wx.Bitmap.FromBufferRGBA(
                w, h, pil_image.convert("RGBA").tobytes())
            dc = wx.ClientDC(self.source_video_panel)
            dc.Clear()
            dc.DrawBitmap(frame, 0, 0, True)

        if euler_angles is not None and self.wx_source_image is not None:
            self.detected_pose = torch.zeros(6, device=self.device)
            self.detected_pose[0] = max(min(-euler_angles.item(0) / 15.0, 1.0), -1.0)
            self.detected_pose[1] = max(min(euler_angles.item(1) / 15.0, 1.0), -1.0)
            self.detected_pose[2] = max(min(-euler_angles.item(2) / 15.0, 1.0), -1.0)

            if self.detected_last_pose is None:
                self.detected_last_pose = self.detected_pose
            else:
                self.detected_pose = self.detected_pose * 0.5 + self.detected_last_pose * 0.5
                self.detected_last_pose = self.detected_pose

            eye_min_ratio = 0.15
            eye_max_ratio = 0.25
            left_eye_normalized_ratio = compute_left_eye_normalized_ratio(face_landmarks, eye_min_ratio, eye_max_ratio)
            self.detected_pose[3] = 1 - left_eye_normalized_ratio
            right_eye_normalized_ratio = compute_right_eye_normalized_ratio(face_landmarks,
                                                                            eye_min_ratio,
                                                                            eye_max_ratio)
            self.detected_pose[4] = 1 - right_eye_normalized_ratio

            min_mouth_ratio = 0.02
            max_mouth_ratio = 0.3
            mouth_normalized_ratio = compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio)
            self.detected_pose[5] = mouth_normalized_ratio

            # self.detected_pose = self.detected_pose.unsqueeze(dim=0)

        current_pose = self.get_current_pose()
        if self.last_pose is not None \
                and self.last_pose == current_pose \
                and self.last_output_index == self.output_index_choice.GetSelection():
            return
        self.last_pose = current_pose
        self.last_output_index = self.output_index_choice.GetSelection()

        if self.torch_source_image is None:
            self.draw_source_image_string(self.result_image_panel, use_paint_dc=False)
            return

        pose = torch.tensor(current_pose, device=self.device)
        output_index = self.output_index_choice.GetSelection()
        output_image = self.poser.pose(self.torch_source_image, pose, output_index)[0].detach().cpu()
        numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
        self.last_output_numpy_image = numpy_image
        wx_image = wx.ImageFromBuffer(
            numpy_image.shape[0],
            numpy_image.shape[1],
            numpy_image[:, :, 0:3].tobytes(),
            numpy_image[:, :, 3].tobytes())
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.ClientDC(self.result_image_panel)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap, (256 - numpy_image.shape[0]) // 2, (256 - numpy_image.shape[1]) // 2, True)

    def on_save_image(self, event: wx.Event):
        if self.last_output_numpy_image is None:
            logging.info("There is no output image to save!!!")
            return

        dir_name = "data/illust"
        file_dialog = wx.FileDialog(self, "Save image", dir_name, "", "*.png", wx.FD_SAVE)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            try:
                if os.path.exists(image_file_name):
                    message_dialog = wx.MessageDialog(self, f"Override {image_file_name}", "Manual Poser",
                                                      wx.YES_NO | wx.ICON_QUESTION)
                    result = message_dialog.ShowModal()
                    if result == wx.ID_YES:
                        self.save_last_numpy_image(image_file_name)
                    message_dialog.Destroy()
                else:
                    self.save_last_numpy_image(image_file_name)
            except:
                message_dialog = wx.MessageDialog(self, f"Could not save {image_file_name}", "Manual Poser", wx.OK)
                message_dialog.ShowModal()
                message_dialog.Destroy()
        file_dialog.Destroy()

    def save_last_numpy_image(self, image_file_name):
        numpy_image = self.last_output_numpy_image
        pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
        os.makedirs(os.path.dirname(image_file_name), exist_ok=True)
        pil_image.save(image_file_name)

    def draw_face_box(self, frame, face_box_points):
        line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        for start, end in line_pairs:
            p0 = (int(face_box_points[start][0]), int(face_box_points[start][1]))
            p1 = (int(face_box_points[end][0]), int(face_box_points[end][1]))
            cv2.line(frame, p0, p1, (255, 0, 0), thickness=2)

    def draw_face_landmarks(self, frame, face_landmarks):
        for i in range(68):
            part = face_landmarks.part(i)
            x = int(part.x)
            y = int(part.y)
            cv2.rectangle(frame, (x - 1, y - 1), (x + 1, y + 1),
                          (0, 255, 0), thickness=2)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    import tha2.poser.modes.mode_20

    poser = tha2.poser.modes.mode_20.create_poser(cuda)

    face_detector = dlib.get_frontal_face_detector()
    landmark_locator = dlib.shape_predictor(
        "data/shape_predictor_68_face_landmarks.dat")

    video_capture = cv2.VideoCapture(0)

    app = wx.App()
    main_frame = MainFrame(poser, face_detector, landmark_locator, video_capture, cuda)
    main_frame.Show(True)
    main_frame.timer.Start(200)
    app.MainLoop()
